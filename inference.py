import argparse
import os
import torch
import numpy as np
from torchvision import transforms as T
import cv2
from collections import deque
import time

from model import CNNLSTM


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt


def sample_indices(total, target):
    if total <= 0:
        return []
    if total >= target:
        return [int(x) for x in np.linspace(0, total - 1, target)]
    idxs = [int(x) for x in np.linspace(0, total - 1, total)]
    pad = [idxs[-1]] * (target - total)
    return idxs + pad


def draw_emotion_text(frame, emotion, confidence, x, y, w, h):
    """Draw emotion text and bounding box on frame"""
    # Draw face bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Prepare text background
    text = f"{emotion}: {confidence:.2f}"
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    
    # Draw text background rectangle
    cv2.rectangle(
        frame,
        (x, y - text_height - 10),
        (x + text_width, y),
        (0, 255, 0),
        -1  # Filled rectangle
    )
    
    # Draw emotion text
    cv2.putText(
        frame,
        text,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),  # Black text
        2
    )
    
    return frame


def extract_frame_features(frame, transform, face_cascade=None, grayscale=False):
    """Extract and preprocess a single frame"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Face detection
    if face_cascade is not None:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            frame_rgb = frame_rgb[y:y+h, x:x+w]
    
    # Apply transformation
    return transform(frame_rgb)


def main():
    parser = argparse.ArgumentParser(description="Real-time emotion detection from video using CNN+LSTM")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=32, help="Number of frames in each analysis window")
    parser.add_argument("--stride", type=int, default=16, help="Stride between analysis windows (frames)")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--use-face", action="store_true")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--fps", type=int, default=30, help="Display FPS")
    parser.add_argument("--emotion-history", type=int, default=10, help="Number of recent emotions to track")
    args = parser.parse_args()

    # Load model and checkpoint
    ckpt = load_checkpoint(args.checkpoint)
    emotion_names = ckpt.get("emotion_names", ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"])
    
    print(f"Emotion classes: {emotion_names}")

    model = CNNLSTM(
        num_classes=len(emotion_names),
        feature_dim=512,
        lstm_hidden=256,
        lstm_layers=1,
        bidirectional=False,
        dropout=0.3,
        pretrained_cnn=False,
        freeze_cnn=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Setup transformations
    if args.grayscale:
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize(args.image_size),
            T.CenterCrop(args.image_size),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
        channels = 1
    else:
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize(args.image_size),
            T.CenterCrop(args.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        channels = 3

    # Initialize face detector
    face_cascade = None
    if args.use_face:
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            print("Face detector initialized successfully")
        except Exception as e:
            print(f"Failed to load face detector: {e}")
            face_cascade = None

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Info: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    print(f"Analysis window: {args.window_size} frames, Stride: {args.stride} frames")
    
    # Frame buffer for analysis window
    frame_buffer = deque(maxlen=args.window_size)
    frame_count = 0
    paused = False
    
    # Track emotion history
    emotion_history = []
    confidence_history = []
    
    # Performance tracking
    processing_times = []
    
    # Create display window
    cv2.namedWindow("Real-time Emotion Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-time Emotion Detection", width, height)
    
    print("\nControls:")
    print("  Q or ESC: Quit")
    print("  P: Pause/Resume")
    print("  Space: Step frame by frame (when paused)")
    print("  R: Reset analysis")
    print("\nStarting analysis...")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Video ended")
                break
            
            frame_count += 1
            
            start_time = time.time()
            
            # Process frame for model
            frame_tensor = extract_frame_features(frame, transform, face_cascade, args.grayscale)
            frame_buffer.append(frame_tensor)
            
            # Analyze emotion when we have enough frames
            current_emotion = "Processing..."
            current_confidence = 0.0
            
            if len(frame_buffer) >= args.window_size:
                # Prepare input tensor
                x = torch.stack(list(frame_buffer), dim=0).unsqueeze(0).to(device)  # [1, T, C, H, W]
                
                with torch.no_grad():
                    logits = model(x)
                    probabilities = torch.softmax(logits, dim=1)
                    pred_idx = torch.argmax(logits, dim=1).item()
                    confidence = probabilities[0, pred_idx].item()
                    
                    current_emotion = emotion_names[pred_idx]
                    current_confidence = confidence
                    
                    # Store history
                    emotion_history.append(current_emotion)
                    confidence_history.append(confidence)
                    
                    # Keep only recent history
                    if len(emotion_history) > args.emotion_history:
                        emotion_history.pop(0)
                        confidence_history.pop(0)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Detect faces in display frame
            if face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    # Draw on all detected faces
                    for (x, y, w, h) in faces:
                        if current_emotion != "Processing...":
                            frame = draw_emotion_text(frame, current_emotion, current_confidence, x, y, w, h)
                        else:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(
                                frame,
                                "Analyzing...",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2
                            )
            
            # Draw emotion information at top
            y_offset = 40
            cv2.putText(
                frame,
                f"Current Emotion: {current_emotion}",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255) if current_emotion != "Processing..." else (0, 165, 255),
                2
            )
            
            if current_emotion != "Processing...":
                cv2.putText(
                    frame,
                    f"Confidence: {current_confidence:.2f}",
                    (20, y_offset + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            
            # Draw emotion history
            if emotion_history:
                y_start = height - 150
                cv2.putText(
                    frame,
                    "Recent Emotions:",
                    (20, y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
                
                # Show most recent 5 emotions
                for i, (emo, conf) in enumerate(zip(emotion_history[-5:], confidence_history[-5:])):
                    y_pos = y_start + (i * 20)
                    cv2.putText(
                        frame,
                        f"{emo}: {conf:.2f}",
                        (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1
                    )
            
            # Draw frame counter and buffer status
            buffer_fill = len(frame_buffer) / args.window_size
            buffer_color = (0, int(255 * buffer_fill), int(255 * (1 - buffer_fill)))
            
            cv2.putText(
                frame,
                f"Frame: {frame_count}/{total_frames}",
                (20, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            cv2.putText(
                frame,
                f"Buffer: {len(frame_buffer)}/{args.window_size}",
                (width - 200, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                buffer_color,
                2
            )
            
            # Draw processing time
            if processing_times:
                avg_time = np.mean(processing_times[-10:])
                cv2.putText(
                    frame,
                    f"Proc: {avg_time*1000:.1f}ms",
                    (width - 200, height - 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 0),
                    1
                )
            
            # Add controls hint
            cv2.putText(
                frame,
                "Q:Quit | P:Pause | Space:Step | R:Reset",
                (width - 350, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )
            
            # Show frame
            cv2.imshow("Real-time Emotion Detection", frame)
            
            # Control playback speed
            delay = max(1, int(1000 / args.fps) - int(avg_time * 1000) if processing_times else int(1000 / args.fps))
            key = cv2.waitKey(delay) & 0xFF
        
        else:
            # When paused, wait for key press
            key = cv2.waitKey(0) & 0xFF
        
        # Handle keyboard controls
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('p'):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        elif key == ord(' ') and paused:
            # Step frame by frame when paused
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            cv2.imshow("Real-time Emotion Detection", frame)
        elif key == ord('r'):
            # Reset analysis
            frame_buffer.clear()
            emotion_history.clear()
            confidence_history.clear()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            print("Analysis reset")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary statistics
    if emotion_history:
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        
        # Count emotions
        from collections import Counter
        emotion_counts = Counter(emotion_history)
        
        print("\nEmotion Distribution:")
        for emotion in emotion_names:
            count = emotion_counts.get(emotion, 0)
            percentage = (count / len(emotion_history)) * 100
            print(f"  {emotion:10s}: {count:3d} frames ({percentage:5.1f}%)")
        
        # Most common emotion
        most_common = emotion_counts.most_common(1)[0] if emotion_counts else ("None", 0)
        print(f"\nMost frequent emotion: {most_common[0]} ({most_common[1]} frames)")
        
        # Average confidence
        avg_confidence = np.mean(confidence_history) if confidence_history else 0
        print(f"Average confidence: {avg_confidence:.3f}")
        
        # Performance stats
        if processing_times:
            avg_processing = np.mean(processing_times) * 1000
            print(f"Average processing time: {avg_processing:.1f}ms per frame")
    
    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()