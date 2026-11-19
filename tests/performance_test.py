"""Performance benchmarking for DeepGuard AI"""
import time
import json
import os
from pathlib import Path
from app.core.orchestrator import DetectionOrchestrator


def benchmark_file(file_path: str, runs: int = 5) -> dict:
    """Benchmark analysis time for a file"""
    orchestrator = DetectionOrchestrator()
    times = []
    
    for i in range(runs):
        start = time.time()
        result = orchestrator.analyze(file_path)
        elapsed = time.time() - start
        times.append(elapsed)
        
        # Verify integrity after each run
        assert result["integrity"]["integrity_preserved"], "File was modified!"
    
    return {
        "file": os.path.basename(file_path),
        "file_size_mb": round(os.path.getsize(file_path) / (1024*1024), 2),
        "runs": runs,
        "avg_time_seconds": round(sum(times) / len(times), 3),
        "min_time_seconds": round(min(times), 3),
        "max_time_seconds": round(max(times), 3),
        "risk_score": result["overall_risk"]["score"],
        "risk_level": result["overall_risk"]["level"]
    }


def main():
    """Run performance benchmarks"""
    test_files = {
        "email": ["samples/phishing_email.eml", "samples/legit_email.eml"],
        "image": ["samples/deepfake_image.jpg", "samples/real_image.jpg"],
        "audio": ["samples/deepfake_audio.wav", "samples/real_audio.wav"],
        "video": ["samples/deepfake_video.mp4", "samples/real_video.mp4"]
    }
    
    results = []
    
    for file_type, files in test_files.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking {file_type.upper()} files")
        print(f"{'='*60}")
        
        for file_path in files:
            if os.path.exists(file_path):
                result = benchmark_file(file_path)
                results.append(result)
                print(f"\n📄 {result['file']}")
                print(f"   Size: {result['file_size_mb']} MB")
                print(f"   Avg Time: {result['avg_time_seconds']}s")
                print(f"   Risk: {result['risk_level']} ({result['risk_score']:.1%})")
            else:
                print(f"⚠️  Skipping {file_path} (not found)")
    
    # Save results
    with open("data/performance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to data/performance_results.json")


if __name__ == "__main__":
    main()