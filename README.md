# Multi-Class Canine Recognition via YOLOv6 Fine-Tuning on Stanford Dogs Dataset

Fine-tuning the YOLOv6 object detection model using the Stanford Dogs Dataset (http://vision.stanford.edu/aditya86/ImageNetDogs/)

## Dataset
- Number of categories: 120
- Number of images: 20,580
- Annotations: Class labels, Bounding boxes
- Split: 70% training (14,406 images), 15% validation (3,087 images), 15% testing (3,087 images)

## Training

```bash
python /content/YOLOv6/tools/train.py \
  --batch-size 32 \
  --conf-file /content/YOLOv6/configs/yolov6n_finetune.py \
  --epochs 100 \
  --img-size 1280 \
  --data-path /content/data.yaml \
  --device 0 \
  --name yolov6n_finetune
```

## Evaluation

```bash
python /content/YOLOv6/tools/eval.py \
  --weights yolov6n.pt \
  --data /content/data.yaml \
  --device 0
```

## Results

### Fine-tuned Model Performance

**Detection Metrics:**
- **mAP@0.50:0.95: 0.691** - Strong overall performance across IoU thresholds
- **mAP@0.50: 0.779** - Excellent detection at relaxed IoU threshold
- **mAP@0.75: 0.739** - Good precision at stricter localization
- **Average Recall (AR)@0.50:0.95: 0.910** - Model detects 91% of ground truth objects

**Performance by Object Size:**
- Small objects: N/A (no small objects in dataset)
- Medium objects: AP 0.555, AR 0.783
- Large objects: AP 0.696, AR 0.913 (best performance)

### Baseline Model Comparison

Evaluation of pre-trained COCO weights without fine-tuning:
- **mAP@0.50:0.95: 0.001** - Essentially zero performance
- **mAP@0.50: 0.001** - Model makes virtually no correct detections
- **Average Recall: 0.103** - Only detects 10% of objects
- Inference time: 0.71 ms (similar speed, but ineffective)

### Key Findings

**Transfer Learning Impact:**
- **691x improvement** in mAP@0.50:0.95 (0.001 → 0.691)
- **8.8x improvement** in recall (0.103 → 0.910)
- Successful adaptation to fine-grained 120-class dog breed classification
