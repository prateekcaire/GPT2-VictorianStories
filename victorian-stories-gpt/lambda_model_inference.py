import json
import boto3
from datetime import datetime
import os

sagemaker_runtime = boto3.client('sagemaker-runtime')

ENDPOINT_NAME = "stem-gpt2-serverless-endpoint"


def lambda_handler(event, context):
    print(f"Lambda handler invoked at {datetime.now()}")
    print(f"Event: {json.dumps(event)}")

    try:
        if 'requestContext' in event and 'connectionId' in event['requestContext']:
            # WebSocket invocation
            connection_id = event['requestContext']['connectionId']
            domain_name = event['requestContext']['domainName']
            stage = event['requestContext']['stage']
            api_client = boto3.client('apigatewaymanagementapi',
                                      endpoint_url=f"https://{domain_name}/{stage}")
            body = json.loads(event['body'])
        else:
            connection_id = None
            api_client = None
            body = event

        action = body.get('action')

        if action == 'sendmessage':
            prompt = body['prompt']
            max_tokens = body.get('max_tokens', 100)

            print(f"Received prompt: {prompt}")
            print(f"Max tokens: {max_tokens}")

            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='application/json',
                Body=json.dumps({
                    'prompt': prompt,
                    'max_tokens': max_tokens
                })
            )

            response_body = json.loads(response['Body'].read().decode())
            generated_text = response_body['generated_text']

            if api_client and connection_id:
                # If it's a WebSocket connection, send tokens individually
                for token in generated_text.split():
                    try:
                        api_client.post_to_connection(
                            ConnectionId=connection_id,
                            Data=json.dumps({'token': token})
                        )
                    except Exception as e:
                        print(f"Error sending token: {str(e)}")

                api_client.post_to_connection(
                    ConnectionId=connection_id,
                    Data=json.dumps({'completion': True})
                )
                return {'statusCode': 200, 'body': json.dumps('Text generation completed')}
            else:
                return {'statusCode': 200, 'body': json.dumps({'generated_text': generated_text})}
        else:
            return {'statusCode': 400, 'body': json.dumps('Invalid action')}

    except Exception as e:
        print(f"Error in lambda_handler: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
