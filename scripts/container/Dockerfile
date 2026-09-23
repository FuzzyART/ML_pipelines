#FROM rocm/tensorflow:rocm7.1.1-py3.12-tf2.18-dev
FROM rocm/pytorch

RUN apt-get update --fix-missing
RUN apt-get install -y fish
RUN pip install scikit-learn ipykernel umap-learn


CMD [/bin/fish]

