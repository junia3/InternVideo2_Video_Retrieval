import torch
from internvideo2_stage2_1b import InternVideo2_S2_1B
from msrvtt_dataset import MSRVTTDataset, collate_func
from tqdm import tqdm
from torch.cuda.amp import autocast
from utils import *

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = InternVideo2_S2_1B(pretrained_path="path", device=device)
    model.compile()

    # Load the dataset
    data_dir = "datadir"
    dataset = MSRVTTDataset(data_dir=data_dir, split='val_list_jsfusion')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_func, num_workers=4)

    video_ids = []
    caption2video = []
    video_outputs = []
    text_outputs = []
    text_attns = []

    video_embeddings = []
    text_embeddings = []
    bar = tqdm(dataloader, desc='Computing embeddings')
    tokenizer = model.tokenizer

    model.eval()
    with torch.no_grad(), autocast(enabled=True, dtype=torch.bfloat16):
        for data in bar:
            video_id, video_paths, caption_list = data["video_ids"], data["video_paths"], data["captions"]
            video_inputs = process_video_frames_batch(video_paths, num_frames=8).to(device)
            text_input = tokenizer(caption_list,
                                    max_length=40,
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt").to(device)
            
            video_output, pooled_video_output = model.encode_vision(video_inputs)
            text_output, pooled_text_output = model.encode_text(text_input)

            video_outputs.append(video_output.detach().cpu())
            text_outputs.append(text_output.detach().cpu())
            text_attns.append(text_input.attention_mask.detach().cpu())
            if video_id:
                for vid in video_id:
                    if not video_ids:
                        video_ids.append(vid)
                    else:
                        if video_ids[-1] != vid:
                            video_ids.append(vid)

            for vid, caption in zip(video_id, caption_list):
                caption2video.append((caption, vid))
            video_embeddings.append(model.vision_proj(pooled_video_output).detach().cpu())
            text_embeddings.append(model.text_proj(pooled_text_output).detach().cpu())

    video_outputs = torch.cat(video_outputs, dim=0)
    text_outputs = torch.cat(text_outputs, dim=0)
    text_attns = torch.cat(text_attns, dim=0)

    video_embeddings = torch.cat(video_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)

    video_embeddings = F.normalize(video_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    id2index = {video_id: i for i, video_id in enumerate(video_ids)}
    sim_matrix = text_embeddings @ video_embeddings.T

    ranks = []
    for i in range(sim_matrix.shape[0]):
        gt_video_id = caption2video[i][1]
        gt_index = id2index[gt_video_id]
        similarity_scores = sim_matrix[i]
        sorted_indices = torch.argsort(similarity_scores, descending=True)
        rank = torch.where(sorted_indices == gt_index)[0][0]
        ranks.append(rank)

    metrics = compute_metrics(ranks)
    print(metrics)

    text_encoder = model.get_text_encoder()
    text_encoder.eval()

    match_head = model.itm_head
    topk_value = 64
    t2i_scores_x = torch.full((sim_matrix.shape[0], sim_matrix.shape[1]), -100.0)
    bar = tqdm(range(sim_matrix.shape[0]), desc='Computing scores')
    for i in bar:
        _, topk_idx = sim_matrix[i].topk(k=topk_value, dim=0)
        bs = min(topk_value, 32)
        itm_embeds = []
        for j in range(0, len(topk_idx), bs):
            encoder_output = (
                video_outputs[topk_idx[j : j+bs]]
            )
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long)
            repeat_n = encoder_output.size(0)
            with torch.no_grad(), autocast(enabled=True, dtype=torch.bfloat16):
                output = text_encoder(
                    encoder_embeds=text_outputs[i].repeat(repeat_n, 1, 1).to(device),
                    attention_mask=text_attns[i].repeat(repeat_n, 1).to(device),
                    encoder_hidden_states=encoder_output.to(device, non_blocking=True),
                    encoder_attention_mask=encoder_att.to(device, non_blocking=True),
                    return_dict=True,
                    mode="fusion",
                )
                batch_itm_embeds = output.last_hidden_state[:, 0]
                itm_embeds.append(batch_itm_embeds)
            itm_embeds = torch.cat(itm_embeds, dim=0)
            score = match_head(itm_embeds)[:, 1]
        t2i_scores_x[i, topk_idx] = score.detach().cpu().to(t2i_scores_x.dtype)

    t2i_scores_x = t2i_scores_x.softmax(dim=1).detach().float()
    ranks = []
    for i in range(t2i_scores_x.shape[0]):
        gt_video_id = caption2video[i][1]
        gt_index = id2index[gt_video_id]
        similarity_scores = t2i_scores_x[i]
        sorted_indices = torch.argsort(similarity_scores, descending=True)
        rank = torch.where(sorted_indices == gt_index)[0][0]
        ranks.append(rank)

    metrics = compute_metrics(ranks)
    print(metrics)

if __name__ == "__main__":
    main()