import sys
sys.path.append("..")
import Network as net


def test_print_metrics():
    y_pred = [1 , 1 , 1 , 1]
    y = [1 , 1 , 1 , 1]

    assert net.print_metrics(y_pred,y) == 1