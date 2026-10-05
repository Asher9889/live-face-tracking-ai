import sys
import time
import threading
from app.api.run_server import start_api
from app.camera import fetch_cameras, start_camera_threads
from app.camera.tracker_assets import TrackerAssetError
from app.recognition import embedding_store, unknown_embedding_store
from app.api.server import wait_for_api

def main():
    print("\n🚀 Starting Live Face Tracking System\n")

    try:
        # Starting HTTP API server on a seprate thread
        api_thread = threading.Thread(target=start_api, daemon=True)
        api_thread.start()

        # Loading all embeddings first
        embedding_store.load_embeddings()
        unknown_embedding_store.load_unknown_embeddings()
     
        # Starting camera workers
        cameras = fetch_cameras()
        if not cameras:
            print("❌ No cameras found. Exiting.")
            return

        """
        Spawn thread for each camera
        """
        print("\n📷 Starting camera workers...")
        start_camera_threads(cameras)

        while True:
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested") 
        time.sleep(1) 

    except TrackerAssetError as e:
        # Tracking is unusable without the ReID checkpoint, so there is no point
        # running the API or the camera workers. Fail loudly instead.
        print("\n🔥 Fatal error: required tracker model is unavailable\n")
        print(f"{e}\n")
        sys.exit(1)

    except Exception as e:
        print(f"\n🔥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
