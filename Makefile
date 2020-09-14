.PHONY: help lint clean init


#################################################################################
# GLOBALS                                                                       #
#################################################################################
export DOCKER=docker
export BASE_IMAGE_NAME=sac_uav
export BASE_DOCKERFILE=Dockerfile
export JUPYTER_HOST_PORT=8888
export JUPYTER_CONTAINER_PORT=8888
export CONTAINER_NAME=sac_uav-container
#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Build docker image
create-image:
	sudo $(DOCKER) build -t $(BASE_IMAGE_NAME) -f $(BASE_DOCKERFILE) --force-rm --build-arg UID=$(shell id -u) .

## Run docker container
create-container: 
	sudo $(DOCKER) run -it -v $(shell pwd):/home/sac_uav -p $(JUPYTER_HOST_PORT):$(JUPYTER_CONTAINER_PORT) --name $(CONTAINER_NAME) $(BASE_IMAGE_NAME)

## Start docker container. Attach if 	already started
start-container: ## start docker container
	@echo "$$START_DOCKER_CONTAINER" | $(SHELL)
	sudo $(DOCKER) start $(CONTAINER_NAME)
	@echo "Launched $(CONTAINER_NAME)..."
	sudo $(DOCKER) attach $(CONTAINER_NAME)

## Stop active containers
stop-container: ## Spin down active containers
	sudo $(DOCKER) container stop $(CONTAINER_NAME)

## Rm containers
clean-container: ## remove Docker container
	sudo $(DOCKER) rm $(CONTAINER_NAME)

## Train your agent
training:
	xvfb-run ./training.sh


## Evaluate with headless
evaluate:
	./evaluate.sh
## Evaluate with headless
evaluate-container:
	xvfb-run ./evaluate_container.sh
	
#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
# .PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')