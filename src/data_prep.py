"""
Script to prepare dataset for training/evaluation
Handles face cropping, alignment, and normalization
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from src.detect_faces import FaceDetector
from src.utils import (
    load_image, save_image, crop_face, align_face,
    calculate_face_quality
)
from src.logger import log


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    align: bool = True,
    min_quality: float = 0.3,
    output_size: tuple = (112, 112)
):
    """
    Prepare dataset by detecting, cropping, and aligning faces
    
    Args:
        input_dir: Directory containing raw images organized by identity
        output_dir: Directory to save processed faces
        align: Whether to align faces
        min_quality: Minimum face quality threshold
        output_size: Output face image size
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = FaceDetector(model_name="mtcnn", device="cpu")
    
    # Statistics
    stats = {
        "total_images": 0,
        "successful": 0,
        "no_face": 0,
        "low_quality": 0,
        "multiple_faces": 0,
        "identities": {}
    }
    
    # Process each identity directory
    identity_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    for identity_dir in tqdm(identity_dirs, desc="Processing identities"):
        identity_name = identity_dir.name
        identity_output = output_path / identity_name
        identity_output.mkdir(parents=True, exist_ok=True)
        
        identity_stats = {"total": 0, "successful": 0}
        
        # Process images for this identity
        image_files = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))
        
        for image_file in tqdm(image_files, desc=f"  {identity_name}", leave=False):
            stats["total_images"] += 1
            identity_stats["total"] += 1
            
            try:
                # Load image
                image = load_image(str(image_file))
                
                # Detect faces
                boxes, landmarks, confidences = detector.detect(
                    image, return_landmarks=True, return_confidence=True
                )
                
                if len(boxes) == 0:
                    stats["no_face"] += 1
                    log.debug(f"No face detected: {image_file}")
                    continue
                
                if len(boxes) > 1:
                    stats["multiple_faces"] += 1
                    log.debug(f"Multiple faces detected: {image_file}")
                    # Use the largest face
                    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
                    idx = np.argmax(areas)
                    box = boxes[idx]
                    landmark = landmarks[idx] if landmarks is not None else None
                else:
                    box = boxes[0]
                    landmark = landmarks[0] if landmarks is not None else None
                
                # Check face quality
                face = crop_face(image, box, margin=0.1)
                quality = calculate_face_quality(face, box)
                
                if quality < min_quality:
                    stats["low_quality"] += 1
                    log.debug(f"Low quality face: {image_file} (quality={quality:.2f})")
                    continue
                
                # Align face if requested
                if align and landmark is not None:
                    aligned_face = align_face(image, landmark, output_size)
                else:
                    # Just resize
                    aligned_face = cv2.resize(face, output_size)
                
                # Save processed face
                output_file = identity_output / image_file.name
                save_image(aligned_face, str(output_file))
                
                stats["successful"] += 1
                identity_stats["successful"] += 1
                
            except Exception as e:
                log.error(f"Error processing {image_file}: {str(e)}")
        
        stats["identities"][identity_name] = identity_stats
    
    # Save statistics
    stats_file = output_path / "preparation_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("Dataset Preparation Summary")
    print("="*50)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Successful: {stats['successful']}")
    print(f"No face detected: {stats['no_face']}")
    print(f"Low quality: {stats['low_quality']}")
    print(f"Multiple faces: {stats['multiple_faces']}")
    print(f"Success rate: {stats['successful']/stats['total_images']*100:.1f}%")
    print(f"\nTotal identities: {len(stats['identities'])}")
    print(f"Output directory: {output_path}")
    print("="*50)


def create_train_val_split(
    data_dir: str,
    output_dir: str,
    val_ratio: float = 0.2,
    min_images_per_identity: int = 2
):
    """
    Create train/val split from prepared dataset
    
    Args:
        data_dir: Directory with processed faces
        output_dir: Directory to save split dataset
        val_ratio: Validation set ratio
        min_images_per_identity: Minimum images required per identity
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each identity
    for identity_dir in data_path.iterdir():
        if not identity_dir.is_dir():
            continue
        
        identity_name = identity_dir.name
        image_files = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))
        
        if len(image_files) < min_images_per_identity:
            log.warning(f"Skipping {identity_name}: only {len(image_files)} images")
            continue
        
        # Shuffle and split
        np.random.shuffle(image_files)
        n_val = max(1, int(len(image_files) * val_ratio))
        
        val_files = image_files[:n_val]
        train_files = image_files[n_val:]
        
        # Create identity directories
        train_identity_dir = train_dir / identity_name
        val_identity_dir = val_dir / identity_name
        train_identity_dir.mkdir(exist_ok=True)
        val_identity_dir.mkdir(exist_ok=True)
        
        # Copy files (or create symlinks)
        import shutil
        for f in train_files:
            shutil.copy(f, train_identity_dir / f.name)
        for f in val_files:
            shutil.copy(f, val_identity_dir / f.name)
    
    print(f"\nCreated train/val split in {output_path}")
    print(f"Train: {len(list(train_dir.iterdir()))} identities")
    print(f"Val: {len(list(val_dir.iterdir()))} identities")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare face dataset")
    parser.add_argument("--input", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--no-align", action="store_true", help="Skip face alignment")
    parser.add_argument("--min-quality", type=float, default=0.3, help="Minimum face quality")
    parser.add_argument("--size", type=int, default=112, help="Output face size")
    parser.add_argument("--create-split", action="store_true", help="Create train/val split")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    
    args = parser.parse_args()
    
    # Prepare dataset
    prepare_dataset(
        input_dir=args.input,
        output_dir=args.output,
        align=not args.no_align,
        min_quality=args.min_quality,
        output_size=(args.size, args.size)
    )
    
    # Create split if requested
    if args.create_split:
        split_output = Path(args.output).parent / f"{Path(args.output).name}_split"
        create_train_val_split(
            data_dir=args.output,
            output_dir=str(split_output),
            val_ratio=args.val_ratio
        )
