# Vietnamese Visual Speech Recognition with CTC and MCTC

Đề tài xây dựng hệ thống nhận dạng tiếng nói trực quan tiếng Việt trên tập dữ liệu ViCocktail. Project gồm hai mô hình:

- `CTCVSR`: dự đoán trực tiếp chuỗi từ bằng CTC.
- `MCTCVSR`: phân rã âm tiết thành âm đầu, vần và thanh điệu.

Cả hai mô hình sử dụng 3D Stem, ShuffleNetV2 và TCN để trích xuất đặc trưng từ video vùng môi.

## Cài đặt

Yêu cầu Linux, Conda và GPU hỗ trợ CUDA.

```bash
bash setup_linux.sh
conda activate venv
```

ViCocktail được tải tự động từ Hugging Face khi chạy chương trình.

## Tải pretrained model

Tải model `snv1x_tcn2x` từ [Model Zoo](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks#model-zoo), sau đó đặt file tại:

```text
checkpoints/pretrain/lrw_snv1x_tcn2x.pth
```

## Huấn luyện CTCVSR

```bash
python train_word.py \
  --configs config.yaml \
  --epochs 20 \
  --fraction 1.0 \
  --min_frames 20 \
  --max_frames 260 \
  --visual_pretrained checkpoints/pretrain/lrw_snv1x_tcn2x.pth \
  --output_dir checkpoints/CTCVSR
```

## Huấn luyện MCTCVSR

```bash
python train_phoneme.py \
  --configs config.yaml \
  --epochs 20 \
  --fraction 1.0 \
  --min_frames 20 \
  --max_frames 260 \
  --visual_pretrained checkpoints/pretrain/lrw_snv1x_tcn2x.pth \
  --output_dir checkpoints/MCTCVSR
```

## Validation

```bash
python val.py --ckpt checkpoints/CTCVSR/best.pt --split val
python val.py --ckpt checkpoints/MCTCVSR/best.pt --split val
```

Để đánh giá trên tập test, thay `val` bằng `test`.

