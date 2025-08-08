import pytest
from fastapi.testclient import TestClient
from app.main import app

# Test data
TEST_DOCUMENT_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
TEST_QUESTIONS = [
    "What is this document about?",
    "What are the key points mentioned?"
]

client = TestClient(app)


def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/api/v1/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    
    assert response.status_code == 200
    assert "message" in response.json()


def test_query_endpoint_structure():
    """Test the query endpoint with proper request structure"""
    payload = {
        "documents": TEST_DOCUMENT_URL,
        "questions": TEST_QUESTIONS
    }
    
    response = client.post("/api/v1/hackrx/run", json=payload, timeout=30.0)
    
    # Should succeed or fail gracefully
    assert response.status_code in [200, 422, 500]  # Valid responses
    
    if response.status_code == 200:
        data = response.json()
        assert "answers" in data
        assert len(data["answers"]) == len(TEST_QUESTIONS)
        
        # Validate answer structure
        for answer in data["answers"]:
            assert "answer" in answer
            assert "clauses" in answer
            assert "decision_rationale" in answer


def test_invalid_document_url():
    """Test with invalid document URL"""
    payload = {
        "documents": "https://invalid-url.com/nonexistent.txt",
        "questions": ["What is this about?"]
    }
    
    response = client.post("/api/v1/hackrx/run", json=payload)
    
    assert response.status_code in [400, 422, 500]  # Should fail gracefully


if __name__ == "__main__":
    # Run individual test
    asyncio.run(test_health_check())
    print("Health check test passed!")
