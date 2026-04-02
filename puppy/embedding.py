import os
import numpy as np
from typing import List, Union
from langchain_community.embeddings import OpenAIEmbeddings


class OpenAILongerThanContextEmb:
    """
    Embedding function with openai as embedding backend.
    If the input is larger than the context size, the input is split into chunks of size `chunk_size` and embedded separately.
    The final embedding is the average of the embeddings of the chunks.
    Details see: https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    """

    def __init__(
        self,
        openai_api_key: Union[str, None] = None,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 5000,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the Embedding object.

        Args:
            openai_api_key (str): The API key for OpenAI.
            embedding_model (str, optional): The model to use for embedding. Defaults to "text-embedding-3-small".
            chunk_size (int, optional): The maximum number of token to send to openai embedding model at one time. Defaults to 5000.
            verbose (bool, optional): Whether to show progress bar during embedding. Defaults to False.

        Returns:
            None
        """
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        # Use a tiktoken model alias known to older tiktoken versions.
        # (text-embedding-3-* maps to cl100k_base tokenization.)
        tiktoken_model_name = "text-embedding-ada-002"
        self.emb_model = OpenAIEmbeddings(
            model=embedding_model,
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
            chunk_size=chunk_size,
            tiktoken_model_name=tiktoken_model_name,
            show_progress_bar=verbose,
        )

    def _emb(self, text: Union[List[str], str]) -> List[List[float]]:
        """
        Asynchronously performs embedding on a list of text.

        This method calls the `aembed_documents` method of the `emb_model` object to embed the input text.

        Args:
            self: The instance of the class.
            text (List[str]): A list of text to be embedded.

        Returns:
            List[List[float]]: The embeddings of the input text as a list of lists of floats.

        """
        if isinstance(text, str):
            text = [text]
        # Normalize payload to avoid malformed JSON issues from control chars.
        cleaned = []
        for t in text:
            s = str(t).replace("\x00", " ")
            s = "".join(ch for ch in s if (ord(ch) >= 32 or ch in "\n\r\t"))
            s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
            cleaned.append(s)
        try:
            return self.emb_model.embed_documents(texts=cleaned, chunk_size=None)
        except Exception:
            # Fallback to per-item embedding so one bad string does not abort the run.
            dim = self.get_embedding_dimension()
            vectors: List[List[float]] = []
            for s in cleaned:
                try:
                    vec = self.emb_model.embed_documents(texts=[s], chunk_size=None)[0]
                except Exception:
                    safe = s.encode("ascii", "ignore").decode("ascii", "ignore")[:4000]
                    try:
                        vec = self.emb_model.embed_documents(
                            texts=[safe if safe else " "], chunk_size=None
                        )[0]
                    except Exception:
                        vec = [0.0] * dim
                vectors.append(vec)
            return vectors

    def __call__(self, text: Union[List[str], str]) -> np.ndarray:
        """
        Performs embedding on a list of text.

        This method calls the `_emb` method to asynchronously embed the input text using the `emb_model` object.

        Args:
            self: The instance of the class.
            text (List[str]): A list of text to be embedded.

        Returns:
            np.array: The embedding of the input text as a NumPy array.

        """
        return np.array(self._emb(text)).astype("float32")

    def get_embedding_dimension(self):
        """
        Returns the dimension of the embedding.

        This method checks the value of `self.emb_model.model` and returns the corresponding embedding dimension. If the model is not implemented, a `NotImplementedError` is raised.

        Args:
            self: The instance of the class.

        Returns:
            int: The dimension of the embedding.

        Raises:
            NotImplementedError: Raised when the embedding dimension for the specified model is not implemented.

        """
        match self.emb_model.model:
            case "text-embedding-3-small":
                return 1536
            case "text-embedding-3-large":
                return 3072
            case "text-embedding-ada-002":
                return 1536
            case _:
                raise NotImplementedError(
                    f"Embedding dimension for model {self.emb_model.model} not implemented"
                )
