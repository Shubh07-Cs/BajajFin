import requests
import json

# Sample request from the problem statement
url = "http://localhost:8000/api/v1/hackrx/run"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer b9db70cd9a73efc01be3c0f9665880f061d7d725197ac591f546f588f774fe60"
}

payload = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?"
    ]
}

def test_api():
    try:
        print("Testing the LLM-Powered Intelligent Query-Retrieval System...")
        print("="*70)
        
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful!")
            print(f"Number of answers: {len(result.get('answers', []))}")
            print("="*70)
            
            for i, answer_data in enumerate(result.get("answers", []), 1):
                print(f"\nQ{i}: {payload['questions'][i-1]}")
                print(f"A{i}: {answer_data['answer']}")
                print(f"Rationale: {answer_data['decision_rationale']}")
                print(f"Clauses found: {len(answer_data.get('clauses', []))}")
                print("-" * 50)
            
            return True
        else:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_api()
    print("\n" + "="*70)
    print("OVERALL RESULT:", "✅ PASSED" if success else "❌ FAILED")
    print("="*70)
