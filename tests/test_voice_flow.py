import inspect
import unittest

from agentic_rag.core.voice.tts import TTSService
from agentic_rag.entrypoints.rest.routes.chat import chat_voice


class VoiceFlowTests(unittest.TestCase):
    def test_voice_endpoint_accepts_frontend_session_id_field(self):
        parameters = inspect.signature(chat_voice).parameters

        self.assertIn("session_id", parameters)
        self.assertIn("sid", parameters)

    def test_tts_does_not_read_reference_section(self):
        answer = (
            "驱动安装完成后，重新连接 USB 线。\n\n"
            "参考来源：\n- [R1] 00 - AXDL 工具使用指南.pdf"
        )

        cleaned = TTSService._clean_for_tts(answer)

        self.assertIn("驱动安装完成后", cleaned)
        self.assertNotIn("参考来源", cleaned)
        self.assertNotIn("AXDL", cleaned)


if __name__ == "__main__":
    unittest.main()
