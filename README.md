# 🚀 NASABinning

<p align="center">
  <img src="./imgs/social_preview.png" alt="NASABinning Banner" width="600"/>
</p>

Desenvolvido para facilitar a decisão de agrupamento de variáveis numéricas e categóricas.

## Principais recursos

| Recurso | Descrição |
|---------|-----------|
| **Binning supervisionado e não supervisionado** | Estratégias plug-and-play (Optimal Binning, quantílico, largura fixa, k-means). |
| **Monotonicidade opcional** | Respeita tendências crescentes ou decrescentes para facilitar interpretação regulatória. |
| **Diferença mínima de **event rate**** | Evita sobreposição de grupos ao unir automaticamente bins muito semelhantes. |
| **Estabilidade temporal** | Calcula PSI, KS e desvio-padrão por safra; aplica penalizações ou mesclagens conforme limiares. |
| **Otimização com Optuna (opcional)** | Busca corte ótimo maximizando IV e estabilidade, com múltiplas penalizações configuráveis. |
| **Integração scikit-learn** | `NASABinner` implementa ``fit`/`transform``, permitindo uso em `Pipeline`. |
| **Relatórios auditáveis** | Geração de tabelas e gráficos em `.xlsx`, `.json` e Matplotlib para `WoE`, event-rate e estabilidade. |


> ⚠️ Ideal para cientistas de dados que atuam com modelagem de `risco de crédito`, `scorecards` e precisam entregar resultados robustos e rastreáveis para auditorias ou produção.

---



## 📦 Instalação (em breve)

```bash
# futura distribuição PyPI
pip install nasabinning 
```


### Para desenvolvimento, clone o repositório e instale dependências extras:
```bash
git clone https://github.com/seu-usuario/NASABinning.git
cd NASABinning
pip install -e ".[dev]"
```

### Exemplo rápido
```python
import pandas as pd
from nasabinning import NASABinner

# X: DataFrame de features; y: Series binária; time_col: safra (YYYYMM)
binner = NASABinner(
    strategy          = "supervised",
    min_event_rate_diff = 0.03,
    check_stability   = True,
    monotonic         = True,
    use_optuna        = False
)

binner.fit(X, y, time_col="AnoMesReferencia")
X_woe = binner.transform(X, return_woe=True)

# Relatório de estabilidade temporal
binner.plot_event_rate_stability()
binner.save_report("reports/binning_report.xlsx")
```

## 📁 Estrutura de Pastas (resumida)
```bash
nasabinning/
├─ __init__.py
├─ binning_engine.py
├─ refinement.py
├─ temporal_stability.py
├─ metrics.py
├─ reporting.py
├─ compare.py
├─ strategies/
│   ├─ supervised.py
│   └─ unsupervised.py
└─ optuna_optimizer.py
```


## 🛣️ Roadmap

v0.1 — MVP supervisionado (Optimal Binning)

v0.2 — Estratégias não supervisionadas e integração total scikit-learn

v0.3 — Módulo de comparação (compare.py) com relatórios paralelos

v1.0 — Publicação no PyPI e documentação completa

## 🤝 Contribuindo
Abra uma issue descrevendo a proposta de melhoria ou bug.

Crie um fork e uma branch: `git checkout -b` feature/nome.

Execute `pytest` antes de abrir o pull request.

Siga o guia de estilo definido em `docs/contrib_guidelines.md` (em breve).

## 📄 Licença
Distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para detalhes.


## 📬 Contato
Para dúvidas ou sugestões, abra uma issue ou envie e-mail para [nasabinning@seu-dominio.com](mailto:nasabinning@seu-dominio.com).