# Edit Banana sidecar contract

Edit Banana is optional and disabled while `EDIT_BANANA_BASE_URL` is empty. It is only called when a `/fig` request supplies a static image/PDF and explicitly asks to recover editability.

The main service sends `multipart/form-data` to `POST /v1/convert` with one `file`. The sidecar returns either an `application/xml` Draw.io `mxfile`, or JSON containing `drawio_xml`/`drawio_base64` and an optional `vlm_called` boolean.

The GPU worker owns SAM, OCR, OpenCV, optional Pix2Text, model residency, single-GPU concurrency, and deletion of raw/intermediate files. The main FastAPI process does not import the worker environment and does not send content to a multimodal API by default.

Every result remains `canonical=false`, `topology_verified=false`, and `publication_status=needs_human_visual_review`. Raster image cells, OCR labels outside the FigureContract allowlist, and connectors without confirmed source/target endpoints are disclosed in the delivery manifest. When the sidecar is unavailable, `/fig` enters the existing human-decision path instead of fabricating a reconstruction.
