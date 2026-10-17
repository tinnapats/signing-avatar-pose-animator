import os
import unittest
from unittest.mock import patch

os.environ.setdefault('ALLOWED_ORIGINS', 'https://sign-demo.vercel.app')
try:
    from fastapi.testclient import TestClient
except ImportError as exc:
    raise unittest.SkipTest('Install FastAPI and httpx to run cloud deployment tests.') from exc
import cloud_backend as backend


class CloudBackendTests(unittest.TestCase):
    def setUp(self):
        # Do not enter lifespan: tests must never download private data or AI models.
        self.client = TestClient(backend.app)

    def test_vercel_origin_can_post_images(self):
        response = self.client.options('/api/compare_pose', headers={
            'Origin': 'https://sign-demo.vercel.app',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'Content-Type'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['access-control-allow-origin'], 'https://sign-demo.vercel.app')

    def test_other_origin_is_not_allowed(self):
        response = self.client.options('/api/compare_pose', headers={
            'Origin': 'https://unrelated.example', 'Access-Control-Request-Method': 'POST'})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn('access-control-allow-origin', response.headers)

    def test_generation_waits_for_database(self):
        with patch.dict(backend.state, dataset='loading'):
            response = self.client.get('/api/generate_sequence', params={'text': 'you'})
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()['ok'])
        self.assertIn('error', response.json())

    def test_public_requests_cannot_select_arbitrary_files(self):
        with patch.dict(backend.state, dataset='ready'), patch.object(backend, 'build_payload', return_value={'frames': []}) as build:
            response = self.client.get('/api/generate_sequence', params={'text': 'you', 'file': '/etc/passwd', 'width': '999999'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(build.call_args.kwargs['file_tokens'], [])
        self.assertEqual(build.call_args.kwargs['width'], 513)

    def test_busy_requests_are_rejected(self):
        with patch.dict(backend.state, dataset='ready'):
            backend.work_lock.acquire()
            try:
                response = self.client.get('/api/generate_sequence', params={'text': 'you'})
            finally:
                backend.work_lock.release()
        self.assertEqual(response.status_code, 429)

    def test_speech_returns_not_ready_without_loading_model(self):
        with patch.object(backend.CloudSpeech, '_asr_state', 'loading'):
            response = self.client.post('/api/transcribe_wav', content=b'not-wav')
        self.assertEqual(response.status_code, 503)

    def test_invalid_wav_and_oversize_are_rejected(self):
        with patch.object(backend.CloudSpeech, '_asr_state', 'ready'):
            self.assertEqual(self.client.post('/api/transcribe_wav', content=b'not-wav').status_code, 400)
            self.assertEqual(self.client.post('/api/transcribe_wav', content=b'x' * 2_000_001).status_code, 413)

    def test_health_does_not_expose_database_path(self):
        body = self.client.get('/api/health').json()
        self.assertNotIn('dataDir', body)
        self.assertIn('datasetState', body)
