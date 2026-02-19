import threading
import time

from app.camera import fetch_cameras, start_camera_threads
from app.ai import start_batch_processor
# from app.camera.worker import start_camera_threads
# from app.ai.batch_processor import start_batch_processor
# from app.recognition.engine import start_recognition_engine



def main():
    print("\n🚀 Starting Live Face Tracking System\n")

    try:
        # Fetch cameras
        cameras = fetch_cameras()

        if not cameras:
            print("❌ No cameras found. Exiting.")
            return

        # Start camera capture
        print("\n📷 Starting camera workers...")
        start_camera_threads(cameras)

        # Start AI batch processor
        print("🧠 Starting AI batch processor...")
        threading.Thread(
            target=start_batch_processor,
            daemon=True
        ).start()

        # Start recognition engine
        # print("🧩 Starting recognition engine...")
        # threading.Thread(
        #     target=start_recognition_engine,
        #     daemon=True
        # ).start()

        # print("\n✅ System started successfully\n")

        # Keep process alive
        while True:
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested")

    except Exception as e:
        print(f"\n🔥 Fatal error: {e}")


if __name__ == "__main__":
    main()
