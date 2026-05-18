"""
Test Script for BuildHive Chatbot
Tests all API endpoints and demonstrates functionality
"""

import pytest

pytest.skip(
    "This file is an API smoke-test script, not a pytest unit test. "
    "Skipping during test collection.",
    allow_module_level=True,
)

import requests
import json
import sys
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}{Colors.END}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def test_chatbot_health():
    """Test /health endpoint"""
    print_header("Testing Chatbot Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        
        if response.status_code == 200:
            print_success("Health check endpoint is working")
            print(f"  Status: {data.get('status')}")
            print(f"  Modules: {data.get('modules')}")
            return True
        else:
            print_error(f"Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False


def test_chat_query(query: str):
    """Test /chat endpoint with a specific query"""
    print(f"\n{Colors.YELLOW}Query: {query}{Colors.END}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"query": query},
            timeout=10
        )
        data = response.json()
        
        if response.status_code == 200:
            status = data.get('status')
            if status == 'success':
                print_success("Query processed successfully")
                print(f"\n{Colors.BLUE}Answer:{Colors.END}")
                print(f"  {data.get('answer')}")
                
                retrieved = data.get('retrieved_docs', [])
                if retrieved:
                    print(f"\n{Colors.BLUE}Retrieved Documents:{Colors.END}")
                    for i, doc in enumerate(retrieved, 1):
                        relevance = doc.get('relevance', 0)
                        print(f"  {i}. {doc.get('question')} (relevance: {relevance:.3f})")
                return True
            else:
                print_error(f"Query failed: {data.get('answer')}")
                return False
        else:
            print_error(f"Request failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False


def run_all_tests():
    """Run all test queries"""
    print_header("BuildHive Chatbot - API Test Suite")
    
    # Test health check
    if not test_chatbot_health():
        print_error("Cannot proceed - health check failed")
        return False
    
    # Sample test queries
    test_queries = [
        # Buyer queries
        "How do I search for construction materials on BuildHive?",
        
        # Seller queries
        "How do I register as a seller on BuildHive?",
        "How do I list a new product for sale?",
        
        # Freelancer queries
        "What types of services can freelancers offer?",
        
        # General queries
        "What is BuildHive?",
        "What are the main features of BuildHive?",
        
        # Payment queries
        "What payment methods are supported?",
        
        # AI tools
        "How does the material recommendation system work?",
        "What is the cost estimator?",
        
        # Orders
        "How do I track my order?",
    ]
    
    print_header("Running Chat Queries")
    
    passed = 0
    failed = 0
    
    for query in test_queries:
        if test_chat_query(query):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print_header("Test Summary")
    total = passed + failed
    print(f"Total Tests: {total}")
    print_success(f"Passed: {passed}")
    if failed > 0:
        print_error(f"Failed: {failed}")
    
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    return failed == 0


def test_off_topic_query():
    """Test that chatbot refuses off-topic queries"""
    print_header("Testing Off-Topic Query Handling")
    
    off_topic_query = "What is the weather in Lahore?"
    print(f"{Colors.YELLOW}Query: {off_topic_query}{Colors.END}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"query": off_topic_query},
            timeout=10
        )
        data = response.json()
        answer = data.get('answer', '')
        
        if "only assist with BuildHive" in answer or "BuildHive-related" in answer:
            print_success("Chatbot correctly refused off-topic query")
            print(f"  Response: {answer}")
            return True
        else:
            print_error("Chatbot accepted off-topic query (security concern)")
            return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    print(f"{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║          BuildHive Chatbot API Test Suite v1.0            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(Colors.END)
    
    try:
        # Run main tests
        main_passed = run_all_tests()
        
        # Test off-topic handling
        off_topic_passed = test_off_topic_query()
        
        # Final status
        print_header("Final Status")
        if main_passed and off_topic_passed:
            print_success("All tests passed! Chatbot is working correctly.")
            sys.exit(0)
        else:
            print_error("Some tests failed. Please review the output above.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test suite interrupted by user{Colors.END}")
        sys.exit(2)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        sys.exit(1)
