"""Abstract LLM interface: single entry point for the local mock and the real server API."""


class BaseLLM:
    name = "base"

    def ask_parent(self, inst, i, cands):
        """Given an instance, a query node i and candidate upstream nodes (all earlier), return the predicted direct upstream index."""
        raise NotImplementedError

    def describe(self):
        return self.name
