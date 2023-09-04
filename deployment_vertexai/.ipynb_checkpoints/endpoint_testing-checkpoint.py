from typing import List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import json
import time
from tqdm import tqdm
import numpy as np

# Specify the path to your JSON file
file_path = "input.json"

# Open and read the JSON file with test json request
with open(file_path, 'r') as json_file:
    instances = json.load(json_file)['instances']


def predict_custom_trained_model_sample(

    instances: Union[List, List[List]],
    project: str='home-credit-turing-college',
    endpoint_id: str="8444293281784791040",
    location: str = "europe-west2",
    api_endpoint: str = "europe-west2-aiplatform.googleapis.com"
):
    """
    `instances` can be either single instance of type list or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )

    return {"default_preds": response.predictions}


# Measure the response time
all_times = []

# Response time for 100 times
for i in tqdm(range(100)):
    # Start
    t0 = time.time_ns() // 10**6
    # Executing function
    pred = predict_custom_trained_model_sample(instances=instances)
    # End
    t1 = time.time_ns() // 10**6
    # Append time taken to get the prediction in ms
    all_times.append(t1 - t0)

# Print the results
print("Stats of the response time: ")
print("Median: ", np.median(all_times))
print("95th percentile: ", np.quantile(all_times, 0.95))
print("Max: ", np.max(all_times))