Place your menu here before running the ingest job.

**Preferred (accurate cuisine filters + full menu listing):** `menu_items.json`  
See the bundled example for the expected shape (`items[]` with `name`, `price`, `currency`, `cuisine`, `category`).

**Fallback:** `sample_menu.pdf` (chunked PDF with heuristic cuisine tags — less accurate than JSON).

```bash
docker compose --profile ingest run --rm ingest
```
