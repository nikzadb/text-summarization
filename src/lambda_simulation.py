import json
import boto3
import time
from moto import mock_lambda, mock_iam
from typing import Dict, Any, Callable
import zipfile
import io
import base64


class LambdaSimulator:
    def __init__(self):
        self.mock_lambda = mock_lambda()
        self.mock_iam = mock_iam()
        self.lambda_client = None
        self.iam_client = None
        self.function_name = "summarization-benchmark"
        
    def __enter__(self):
        self.mock_lambda.start()
        self.mock_iam.start()
        
        self.lambda_client = boto3.client('lambda', region_name='us-east-1')
        self.iam_client = boto3.client('iam', region_name='us-east-1')
        
        self._create_lambda_role()
        self._create_lambda_function()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mock_lambda.stop()
        self.mock_iam.stop()
    
    def _create_lambda_role(self):
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "lambda.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        self.iam_client.create_role(
            RoleName='lambda-execution-role',
            AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
            Path='/',
        )
    
    def _create_lambda_function(self):
        lambda_code = '''
import json
import time

def lambda_handler(event, context):
    start_time = time.time()
    
    # Simulate processing
    text = event.get('text', '')
    summarizer_type = event.get('summarizer_type', 'textrank')
    max_sentences = event.get('max_sentences', 3)
    
    # Simulate different processing times based on method
    processing_times = {
        'textrank': 0.1,
        'tfidfrank': 0.15,
        'bart': 2.0,
        't5': 1.5,
        'distilbart': 1.0,
        'gemini': 0.5
    }
    
    # Simulate processing delay
    time.sleep(processing_times.get(summarizer_type, 0.1))
    
    end_time = time.time()
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'summary': f'Simulated summary of {len(text)} characters using {summarizer_type}',
            'processing_time': end_time - start_time,
            'method': summarizer_type,
            'lambda_execution': True
        })
    }
'''
        
        # Create a zip file containing the lambda function
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr('lambda_function.py', lambda_code)
        zip_buffer.seek(0)
        
        self.lambda_client.create_function(
            FunctionName=self.function_name,
            Runtime='python3.9',
            Role='arn:aws:iam::123456789012:role/lambda-execution-role',
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': zip_buffer.read()},
            Description='Summarization benchmark function',
            Timeout=300,
            MemorySize=512,
        )
    
    def invoke_lambda(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.lambda_client.invoke(
            FunctionName=self.function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        
        response_payload = json.loads(response['Payload'].read())
        
        return {
            'statusCode': response_payload['statusCode'],
            'body': json.loads(response_payload['body']),
            'execution_duration': response.get('Duration', 0),
            'billed_duration': response.get('BilledDuration', 0),
            'memory_used': response.get('MemoryUsed', 0),
            'max_memory_used': response.get('MaxMemoryUsed', 0)
        }
    
    def benchmark_summarizer_in_lambda(self, text: str, summarizer_type: str, max_sentences: int = 3) -> Dict[str, Any]:
        payload = {
            'text': text,
            'summarizer_type': summarizer_type,
            'max_sentences': max_sentences
        }
        
        start_time = time.time()
        result = self.invoke_lambda(payload)
        end_time = time.time()
        
        return {
            'lambda_result': result,
            'total_time': end_time - start_time,
            'cold_start_overhead': max(0, (end_time - start_time) - result['body'].get('processing_time', 0))
        }