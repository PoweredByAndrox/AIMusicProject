up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

restart:
	docker compose down && docker compose up --build

api-lint:
	python -m py_compile apps/api/app/*.py apps/api/app/routes/*.py apps/api/app/services/*.py

zip:
	cd .. && zip -r stemlab_multicontainer_blueprint.zip stemlab_multicontainer_blueprint
