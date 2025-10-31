"""
Load Testing with Locust
Performance and stress testing for Dual RAG LLM

Author: Adrian Johnson <adrian207@gmail.com>

Usage:
    locust -f tests/load/locustfile.py --host=http://localhost:8000
    locust -f tests/load/locustfile.py --host=http://localhost:8000 --headless -u 100 -r 10 -t 5m
"""

from locust import HttpUser, task, between, tag
import random


class DualRAGUser(HttpUser):
    """Simulated user for load testing"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    # Sample queries for testing
    queries = [
        "How do I use FastAPI with async?",
        "What is the best way to handle errors in Python?",
        "Explain TypeScript interfaces",
        "How to deploy Docker containers?",
        "What are PostgreSQL indexes?",
        "How to use React hooks?",
        "Explain Kubernetes deployments",
        "What is Redis caching?",
        "How to write unit tests in pytest?",
        "What are SQL joins?"
    ]
    
    file_contexts = [
        "app.py",
        "main.ts",
        "docker-compose.yml",
        "test_api.py",
        "README.md"
    ]
    
    @task(10)
    @tag('query')
    def query_basic(self):
        """Basic query request"""
        self.client.post("/query", json={
            "query": random.choice(self.queries),
            "file_context": random.choice(self.file_contexts)
        })
    
    @task(5)
    @tag('query', 'stream')
    def query_stream(self):
        """Streaming query request"""
        self.client.post("/query/stream", json={
            "query": random.choice(self.queries),
            "file_context": random.choice(self.file_contexts)
        }, stream=True)
    
    @task(3)
    @tag('health')
    def health_check(self):
        """Health check endpoint"""
        self.client.get("/health")
    
    @task(2)
    @tag('models')
    def list_models(self):
        """List models endpoint"""
        self.client.get("/models")
    
    @task(1)
    @tag('analytics')
    def get_analytics(self):
        """Get analytics data"""
        self.client.get("/api/analytics/query")
    
    @task(1)
    @tag('cost')
    def get_cost_summary(self):
        """Get cost summary"""
        self.client.get("/api/cost/summary")
    
    def on_start(self):
        """Called when a simulated user starts"""
        # Could add login or initialization here
        pass


class AdminUser(HttpUser):
    """Simulated admin user for management endpoints"""
    
    wait_time = between(5, 10)
    
    @task(5)
    @tag('admin', 'analytics')
    def get_analytics_report(self):
        """Get comprehensive analytics report"""
        self.client.get("/api/analytics/report")
    
    @task(3)
    @tag('admin', 'audit')
    def get_audit_logs(self):
        """Get audit logs"""
        self.client.get("/api/audit/logs", params={"limit": 50})
    
    @task(2)
    @tag('admin', 'backup')
    def list_backups(self):
        """List backups"""
        self.client.get("/api/backup/list")
    
    @task(2)
    @tag('admin', 'dr')
    def check_dr_status(self):
        """Check disaster recovery status"""
        self.client.get("/api/dr/status")
    
    @task(1)
    @tag('admin', 'encryption')
    def check_encryption_status(self):
        """Check encryption status"""
        self.client.get("/api/encryption/status")


class StressTestUser(HttpUser):
    """Heavy load user for stress testing"""
    
    wait_time = between(0.1, 0.5)  # Very short wait time
    
    @task
    def rapid_fire_queries(self):
        """Rapid queries to stress test the system"""
        for _ in range(5):
            self.client.post("/query", json={
                "query": "Quick test query",
                "file_context": "test.py"
            })

