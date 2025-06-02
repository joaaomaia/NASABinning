import pandas as pd
import numpy as np
from nasabinning.temporal_stability import event_rate_by_time, psi_over_time

def test_event_rate_pivot_and_psi():
    # cria tabela fake
    data = []
    for safra in [202301, 202302]:
        for b in range(3):
            data.append(
                dict(variable="x", bin=b, event=np.random.randint(10, 30),
                     count=100, AnoMesReferencia=safra)
            )
    df = pd.DataFrame(data)
    pivot = event_rate_by_time(df, "AnoMesReferencia")
    assert pivot.shape == (3, 2)
    psi = psi_over_time(pivot)
    assert psi >= 0
