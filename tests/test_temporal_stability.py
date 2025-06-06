import pandas as pd
import numpy as np
from nasabinning.temporal_stability import (
    event_rate_by_time,
    psi_over_time,
    temporal_separability_score,
)

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


def test_temporal_separability_score():
    df = pd.DataFrame({
        'bin': [0, 0, 1, 1] * 3,
        'target': [0, 1, 0, 1] * 3,
        'time': [202301] * 4 + [202302] * 4 + [202303] * 4,
    })
    score = temporal_separability_score(
        df, 'x', 'bin', 'target', 'time'
    )
    assert score > 0
