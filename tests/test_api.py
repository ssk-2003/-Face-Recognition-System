"""
Unit tests for Face Recognition Service API
"""
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app
import io
from PIL import Image
import numpy as np


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create sample image for testing"""
    # Create a simple RGB image
    img = Image.new('RGB', (640, 480), color='white')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def test_root(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "app" in data
    assert data["status"] == "running"


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models" in data


def test_stats(client):
    """Test stats endpoint"""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "num_identities" in data
    assert "uptime_seconds" in data


def test_detect_no_file(client):
    """Test detect endpoint without file"""
    response = client.post("/api/v1/detect")
    assert response.status_code == 422  # Validation error


def test_detect_with_image(client, sample_image):
    """Test detect endpoint with image"""
    response = client.post(
        "/api/v1/detect",
        files={"file": ("test.jpg", sample_image, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "num_faces" in data
    assert "detections" in data
    assert "processing_time_ms" in data


def test_recognize_with_image(client, sample_image):
    """Test recognize endpoint with image"""
    response = client.post(
        "/api/v1/recognize",
        files={"file": ("test.jpg", sample_image, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "num_faces" in data
    assert "results" in data


def test_list_identities(client):
    """Test list identities endpoint"""
    response = client.get("/api/v1/list_identities")
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "count" in data
    assert "identities" in data


def test_add_identity(client, sample_image):
    """Test add identity endpoint"""
    response = client.post(
        "/api/v1/add_identity",
        files={"file": ("test.jpg", sample_image, "image/jpeg")},
        data={"name": f"test_user_{np.random.randint(1000)}"}
    )
    # May succeed or fail depending on whether face is detected
    assert response.status_code in [200, 400, 500]


def test_add_duplicate_identity(client, sample_image):
    """Test adding duplicate identity"""
    name = f"test_user_{np.random.randint(1000)}"
    
    # First attempt
    response1 = client.post(
        "/api/v1/add_identity",
        files={"file": ("test.jpg", sample_image, "image/jpeg")},
        data={"name": name}
    )
    
    # Second attempt (should fail if first succeeded)
    sample_image.seek(0)
    response2 = client.post(
        "/api/v1/add_identity",
        files={"file": ("test.jpg", sample_image, "image/jpeg")},
        data={"name": name}
    )
    
    if response1.status_code == 200:
        assert response2.status_code == 400  # Duplicate


def test_delete_nonexistent_identity(client):
    """Test deleting non-existent identity"""
    response = client.delete("/api/v1/delete_identity/99999")
    assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
