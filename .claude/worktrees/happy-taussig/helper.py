import matplotlib.pyplot as plt

try:
    from IPython import display as _ipython_display
    _HAS_IPYTHON = True
except ImportError:
    _HAS_IPYTHON = False

plt.ion()


def plot(scores, mean_scores):
    """Render or refresh the live training-progress plot."""
    if not scores or not mean_scores:
        return

    if _HAS_IPYTHON:
        _ipython_display.clear_output(wait=True)
        _ipython_display.display(plt.gcf())

    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores,      label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.ylim(ymin=0)
    plt.legend()
    plt.text(len(scores) - 1,      scores[-1],      str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], f'{mean_scores[-1]:.2f}')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
