from queue import Queue, Full, Empty
from typing import Optional
from .types import FrameMessage

# Max frames buffer (tune later)
MAX_QUEUE_SIZE = 250

class FrameQueue:
    """
    Thread-safe queue for transporting frames between
    camera capture and AI processing.
    """

    def __init__(self, max_size: int = MAX_QUEUE_SIZE):
        self._queue = Queue(maxsize=max_size)

    def push(self, frame_msg: FrameMessage) -> None:
        """
        Push frame into queue.
        If full → drop oldest frame to keep system realtime.
        """
 
        try:
            self._queue.put_nowait(frame_msg)

        except Full:
            try:
                # Drop oldest
                self._queue.get_nowait()
            except Empty:
                pass

        # Push again
        self._queue.put_nowait(frame_msg)
    
    def pop(self, timeout: float = 0.1) -> Optional[FrameMessage]:
        """
        Pop frame from queue.
        Returns None if empty.
        """

        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None
            
    def size(self) -> int:
        return self._queue.qsize()

frame_queue = FrameQueue()