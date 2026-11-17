import json

TOOL_MAKERS = ("<tool_call>", "<function=", "<tool_response>")

class HelperStream:

    def _sse_delta(self, response_id: str, text: str)-> str:
        chunk = {
            "id": response_id,
            "model": self.model_name,
            "choices": [
                {
                    "delta": 
                    {
                        "content": text
                    }, 
                    "index": 0
                }
            ],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    @staticmethod
    def _split_safe(buf: str):
        """Riorna (da_inviare, trattenuto, tool_mode)."""

        idxs = [buf.find(m) for m in TOOL_MAKERS if m in buf]

        if idxs:
            i = min(idxs)
            return buf[:i], buf[i:], True
        hold = 0
        for m in TOOL_MAKERS:
            for k in range(min(len(m) - 1, len(buf)), 0, -1):
                if buf.endswith(m[:k]):
                    hold = max(hold, k)
                    break
        return buf[:len(buf) - hold], buf[len(buf) - hold:], False