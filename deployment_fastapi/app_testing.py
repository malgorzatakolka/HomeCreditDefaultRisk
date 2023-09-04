from locust import HttpUser, task, constant_throughput
import json

# Specify the path to your JSON file
file_path = "ex_entry.json"

# Open and read the JSON file with test json request
with open(file_path, 'r') as json_file:
    test_application = json.load(json_file)

class DefaultRisk(HttpUser):
    # Means that a user will send 1 request per second
    wait_time = constant_throughput(1)
    
    # Task to be performed (send data & get response)
    @task
    def predict(self):
        self.client.post(
            "/predict",
            json=test_application,
            timeout=1,
        )