from kubeflow_trainer.components.iris import main as iris

register = dict(
    iris=iris
)

if __name__ == '__main__':
    register["iris"]()
