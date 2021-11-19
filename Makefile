# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt



trainer:
	@ python -m TaxiFareModel.trainer

check_code:
	@flake8 scripts/* TaxiFareModel/*.py

black:
	@black scripts/* TaxiFareModel/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr TaxiFareModel-*.dist-info
	@rm -fr TaxiFareModel/__pycache__*
	@rm -fr TaxiFareModel.egg-info
	@rm -fr mlruns

install:
	@pip install -e . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)



# --------------------------------

#				UPLOAD DATA TO GCP

# -----------------------------------

PROJECT_ID=le-wagon-batch-722
BUCKET_NAME=wagon-data-722-idi
REGION=europe-west2


LOCAL_PATH="raw_data/train_10k.csv"
LOCAL_PATH_2="raw_data/train.csv"

BUCKET_FOLDER ="data"

BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})
BUCKET_FILE_NAME_2=$(shell basename ${LOCAL_PATH_2})

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

upload_data:
	@gsutil cp ${LOCAL_PATH_2} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME_2}



PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=1.15

PACKAGE_NAME=TaxiFareModel
FILENAME=trainer

# will store the packages uploaded to GCP for the training
BUCKET_TRAINING_FOLDER = 'trainings'

JOB_NAME=taxi_fare_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')

gcp_submit_training:
	@gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs
