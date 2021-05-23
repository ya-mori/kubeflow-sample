from kubeflow_preprocessor.components.iris import main as iris

register = dict(iris=iris)

if __name__ == "__main__":
    register["iris"]()
