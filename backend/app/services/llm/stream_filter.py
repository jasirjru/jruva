"""Suppress private thinking blocks even when delimiters cross token boundaries."""


class ThinkingFilter:
    def __init__(self):
        self.pending = ""
        self.in_thinking = False

    def feed(self, text: str) -> str:
        self.pending += text
        visible = []
        while self.pending:
            marker = "</think>" if self.in_thinking else "<think>"
            position = self.pending.find(marker)
            if position >= 0:
                if not self.in_thinking:
                    visible.append(self.pending[:position])
                self.pending = self.pending[position + len(marker):]
                self.in_thinking = not self.in_thinking
                continue
            # Keep only a possible delimiter prefix until the next token arrives.
            retained = 0
            for length in range(1, min(len(marker), len(self.pending) + 1)):
                if self.pending.endswith(marker[:length]):
                    retained = length
            ready = self.pending[:-retained] if retained else self.pending
            if not self.in_thinking:
                visible.append(ready)
            self.pending = self.pending[-retained:] if retained else ""
            break
        return "".join(visible)
