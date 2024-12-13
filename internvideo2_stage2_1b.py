import torch
from torch import nn
from models.backbones.internvideo2 import PretrainInternVideo2
from models.backbones.bert.xbert import BertModel, BertConfig
from models.backbones.bert.tokenization_bert import BertTokenizer
import pdb

class InternVideo2_S2_1B(nn.Module):
    def __init__(self, pretrained_path=None, device="cpu"):
        super(InternVideo2_S2_1B, self).__init__()
        try:
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-large-uncased",
                use_local_only=True,
            )
        except:
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-large-uncased",
                use_local_only=False,
            )

        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.text_encoder = self.build_text_encoder()
        self.vision_proj = nn.Linear(768, 512)
        self.text_proj = nn.Linear(1024, 512)
        self.itm_head = nn.Linear(1024, 2)

        if pretrained_path:
            self.load_model(pretrained_path)
        
        self.to(device)
        self.eval()

    def load_model(self, pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        if "module" in checkpoint:
            checkpoint = checkpoint["module"]

        new_checkpoint = {}
        for name, param in checkpoint.items():
            if "bert" in name:
                new_name = name.replace("bert.", "")
                new_checkpoint[new_name] = param
            else:
                new_checkpoint[name] = param
            param.requires_grad = False

        orig_t_size=4
        embed_name = ["vision_encoder.pos_embed", "vision_encoder.clip_pos_embed"]
        for pos_name in embed_name:
            if pos_name in new_checkpoint:
                pos_embed_checkpoint = new_checkpoint[pos_name]
                embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
                num_patches = self.vision_encoder.patch_embed.num_patches # 
                num_extra_tokens = self.vision_encoder.pos_embed.shape[-2] - num_patches # 0/1
                new_t_size = self.vision_encoder.num_frames // self.vision_encoder.tubelet_size
                orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(orig_t_size)) ** 0.5)
                new_size = int((num_patches // (new_t_size))** 0.5)
                
                if orig_t_size != new_t_size:
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.view(1, orig_t_size, -1, embedding_size)
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, embedding_size, orig_t_size)
                    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=new_t_size, mode='linear')
                    pos_tokens = pos_tokens.view(1, -1, embedding_size, new_t_size)
                    pos_tokens = pos_tokens.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    new_checkpoint[pos_name] = new_pos_embed
                    pos_embed_checkpoint = new_pos_embed

                if orig_size != new_size:
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size) 
                    pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    new_checkpoint[pos_name] = new_pos_embed

        self.load_state_dict(new_checkpoint, strict=False)

    @torch.no_grad()
    def encode_vision(self, image):
        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]
        vision_embeds, pooled_vision_embeds, _, _ = self.vision_encoder(
            image, None, use_image)
        return vision_embeds, pooled_vision_embeds

    @torch.no_grad()
    def encode_text(self, text):
        text_output = self.get_text_encoder()(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        pooled_text_embeds = text_embeds[:, 0]
        return text_embeds, pooled_text_embeds

    def build_vision_encoder(self):
        vision_encoder = PretrainInternVideo2(
            in_chans=3, img_size=224, patch_size=14,
            embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
            clip_embed_dim=768,
            attn_pool_num_heads=16, qkv_bias=False,
            drop_path_rate=0.25,
            init_values=0.00001,
            qk_normalization=True,
            use_flash_attn=False,
            use_fused_rmsnorm=False,
            use_fused_mlp=False,
            fused_mlp_heuristic=1,
            layerscale_no_force_fp32=False,
            num_frames=8,
            tubelet_size=1,
            sep_pos_embed=False,
            sep_image_video_pos_embed=True,
            use_checkpoint=False,
            checkpoint_num=40,
            clip_teacher_embed_dim=3200,
            clip_teacher_final_dim=768,
            clip_norm_type='l2',
            clip_return_layer=6,
            clip_student_return_interval=1,
        )
        return vision_encoder

    def build_text_encoder(self):
        encoder_name = "bert_large"
        bert_config = BertConfig.from_json_file("configs/config_bert_large.json")
        bert_config.encoder_width = 1408

        bert_config.gradient_checkpointing = False
        try:
            text_encoder, _ = BertModel.from_pretrained(
                "bert-large-uncased",
                config=bert_config,
                add_pooling_layer=False,
                output_loading_info=True,
                local_files_only=True,
            )
        except:
            text_encoder, _ = BertModel.from_pretrained(
                "bert-large-uncased",
                config=bert_config,
                add_pooling_layer=False,
                output_loading_info=True,
                local_files_only=False,
            )
        return text_encoder

    def get_text_encoder(self):
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder