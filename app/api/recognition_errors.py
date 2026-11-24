"""Recognition exception types used by routes for safe, id-only rejection logging.

All HTTP status/code/message mapping lives centrally in app/api/errors.py.
"""
from app.recognition import MultipleFacesError, NoFaceDetectedError, RecognitionUnavailableError

RECOGNITION_ERRORS = (RecognitionUnavailableError, NoFaceDetectedError, MultipleFacesError)
