"""
RAG pipeline implementation using LangGraph.
Combines document retrieval, context generation, and LLM response generation.
"""

import os
from typing import List, Dict, Any, Optional, Union, TypedDict, Annotated
from dotenv import load_dotenv
import json
import logging

from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from RAG.data.qdrant_connector import QdrantConnector
from RAG.models.gemini_interface import GeminiLLM
from RAG.models.gemini_embeddings import GeminiEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define state types
class RAGState(TypedDict):
    """Type for the state of the RAG pipeline."""
    query: str
    image_path: Optional[str]
    retrieved_documents: List[Dict[str, Any]]
    context: str
    response: str
    error: Optional[str]

class RAGPipeline:
    """
    RAG pipeline that combines document retrieval, context generation, and LLM response.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4-vision-preview",
        qdrant_collection: str = "documents",
        num_results: int = 5
    ):
        """
        Initialize the RAG pipeline.

        Args:
            embedding_model: Name of the OpenAI embedding model to use.
            llm_model: Name of the OpenAI LLM model to use.
            qdrant_collection: Name of the Qdrant collection to use.
            num_results: Number of documents to retrieve.
        """
        load_dotenv()

        # Initialize components
        self.embeddings = GeminiEmbeddings(
            model_name="embedding-001",
            task_type="RETRIEVAL_QUERY"
        )
        self.qdrant = QdrantConnector(collection_name=qdrant_collection)
        self.llm = GeminiLLM(model_name=llm_model)
        self.num_results = num_results

        # Build the graph
        self.graph = self._build_graph()

        print(f"RAG pipeline initialized with {embedding_model}, {llm_model}, {num_results} results")

    def _retrieve_documents(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents from Qdrant based on the query.

        Args:
            state: Current state of the RAG pipeline.

        Returns:
            Updated state with retrieved documents.
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(state["query"])

            # Search Qdrant
            results = self.qdrant.search(query_embedding, limit=self.num_results)

            # Update state
            return {
                **state,
                "retrieved_documents": results,
                "error": None
            }
        except Exception as e:
            error_msg = f"Error retrieving documents: {str(e)}"
            print(error_msg)
            return {
                **state,
                "retrieved_documents": [],
                "error": error_msg
            }

    def _generate_context(self, state: RAGState) -> RAGState:
        """
        Generate context from retrieved documents.

        Args:
            state: Current state of the RAG pipeline.

        Returns:
            Updated state with generated context.
        """
        documents = state["retrieved_documents"]

        if not documents:
            return {
                **state,
                "context": "",
                "error": state["error"] or "No documents retrieved"
            }

        # Sort by score (highest first)
        sorted_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)

        # Generate context string
        context_parts = []
        for i, doc in enumerate(sorted_docs):
            title = doc.get("title", "Untitled")
            content = doc.get("content", "")
            url = doc.get("url", "")

            context_part = f"Document {i+1}: {title}\nURL: {url}\nContent: {content}\n"
            context_parts.append(context_part)

        context = "\n".join(context_parts)

        return {
            **state,
            "context": context,
            "error": None
        }

    def _generate_response(self, state: RAGState) -> RAGState:
        """
        Generate response from the LLM using the query, image, and context.

        Args:
            state: Current state of the RAG pipeline.

        Returns:
            Updated state with generated response.
        """
        try:
            # Generate response
            response = self.llm.generate_response(
                prompt=state["query"],
                image_path_or_url=state["image_path"],
                context=state["context"]
            )

            return {
                **state,
                "response": response,
                "error": None
            }
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return {
                **state,
                "response": f"I encountered an error while processing your request: {str(e)}",
                "error": error_msg
            }

    def _should_end(self, state: RAGState) -> bool:
        """
        Determine if the pipeline should end.

        Args:
            state: Current state of the RAG pipeline.

        Returns:
            True if the pipeline should end, False otherwise.
        """
        # End if we have a response or an error
        return bool(state["response"]) or bool(state["error"])

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph for the RAG pipeline.

        Returns:
            StateGraph instance.
        """
        # Create the graph
        graph = StateGraph(RAGState)

        # Add nodes
        graph.add_node("retrieve_documents", self._retrieve_documents)
        graph.add_node("generate_context", self._generate_context)
        graph.add_node("generate_response", self._generate_response)

        # Add edges
        graph.add_edge("retrieve_documents", "generate_context")
        graph.add_edge("generate_context", "generate_response")
        graph.add_edge("generate_response", END)

        # Set entry point
        graph.set_entry_point("retrieve_documents")

        # Compile the graph
        return graph.compile()

    def run(self, query: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the RAG pipeline with the given query and optional image.

        Args:
            query: User query string.
            image_path: Optional path or URL to an image.

        Returns:
            Dictionary with the final state of the pipeline.
        """
        # Initialize state
        initial_state: RAGState = {
            "query": query,
            "image_path": image_path,
            "retrieved_documents": [],
            "context": "",
            "response": "",
            "error": None
        }

        # Run the graph
        try:
            result = self.graph.invoke(initial_state)
            return result
        except Exception as e:
            print(f"Error running RAG pipeline: {e}")
            return {
                **initial_state,
                "error": f"Error running RAG pipeline: {str(e)}"
            }


if __name__ == "__main__":
    # Simple test
    try:
        pipeline = RAGPipeline(num_results=3)

        # Test with text only
        result = pipeline.run(
            query="What is retrieval-augmented generation?",
        )

        print("\nText-only query result:")
        print(f"Query: {result['query']}")
        print(f"Retrieved {len(result['retrieved_documents'])} documents")
        print(f"Response: {result['response']}")

        # Test with image (replace with a valid URL or local path)
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/ChatGPT_logo.svg/1024px-ChatGPT_logo.svg.png"
        image_result = pipeline.run(
            query="What is shown in this image?",
            image_path=image_url
        )

        print("\nImage query result:")
        print(f"Query: {image_result['query']}")
        print(f"Retrieved {len(image_result['retrieved_documents'])} documents")
        print(f"Response: {image_result['response']}")

    except Exception as e:
        print(f"Error in RAG pipeline test: {e}")
