
jupyter:
	poetry run jupyter lab

format:
	poetry run black ./
	poetry run isort ./
