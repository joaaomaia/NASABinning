# 🚀 NASABinning

<p align="center">
  <img src="./imgs/social_preview.png" alt="NASABinning Banner" width="600"/>
</p>

**NASABinning**
biblioteca desenvolvida para facilitar o processo de agrupamento de variáveis numéricas e categóricas.

## Principais recursos

| Recurso | Descrição |
|---------|-----------|
| **Binning supervisionado e não supervisionado** | Estratégias plug-and-play (Optimal Binning, quantílico, largura fixa, k-means). |
| **Monotonicidade opcional** | Respeita tendências crescentes ou decrescentes para facilitar interpretação regulatória. |
| **Diferença mínima de *event rate*** | Evita sobreposição de grupos ao unir automaticamente bins muito semelhantes. |
| **Estabilidade temporal** | Calcula PSI, KS e desvio-padrão por safra; aplica penalizações ou mesclagens conforme limiares. |
| **Otimização com Optuna (opcional)** | Busca corte ótimo maximizando IV e estabilidade, com múltiplas penalizações configuráveis. |
| **Integração scikit-learn** | `NASABinner` implementa `fit/transform`, permitindo uso em `Pipeline`. |
| **Relatórios auditáveis** | Geração de tabelas e gráficos em `.xlsx`, `.json` e Matplotlib para WoE, event-rate e estabilidade. |



> ⚠️ Ideal para cientistas de dados que atuam com modelagem de risco de crédito, scorecards e precisam entregar resultados robustos e rastreáveis para auditorias ou produção.

---



## 📦 Instalação (em breve)

```bash
# futura distribuição PyPI
pip install nasabinning 
