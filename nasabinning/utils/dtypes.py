# nasabinning/utils/dtypes.py
from __future__ import annotations
import warnings, pandas as pd
from typing import List, Optional, Tuple


def search_dtypes(
    df: pd.DataFrame, 
    target_col: str = 'target', 
    limite_categorico: int = 50, 
    force_categorical: Optional[List[str]] = None, 
    verbose: bool = True, 
    remove_ids: bool = False,
    id_patterns: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    Identifica e classifica colunas numéricas e categóricas em um DataFrame.

    Funcionalidades:
    - Valida entradas e trata erros de forma robusta
    - Força colunas específicas como categóricas
    - Classifica automaticamente por tipo de dados e cardinalidade
    - Remove colunas de ID opcionalmente
    - Suporte a padrões customizados para identificação de IDs

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame de entrada para análise
    target_col : str, default 'target'
        Nome da coluna target a ser excluída da análise
    limite_categorico : int, default 50
        Máximo de valores únicos para considerar coluna object como categórica
    force_categorical : List[str], optional
        Lista de colunas que devem ser forçadas como categóricas
    verbose : bool, default True
        Se True, imprime detalhes das decisões tomadas
    remove_ids : bool, default False
        Se True, remove colunas identificadas como IDs
    id_patterns : List[str], optional
        Padrões para identificar colunas de ID (ex: ['_id', 'id_', 'codigo'])

    Retorna:
    --------
    Tuple[List[str], List[str]]
        Tupla contendo (colunas_numericas, colunas_categoricas)

    Raises:
    -------
    ValueError
        Se o DataFrame estiver vazio ou se target_col não existir
    TypeError
        Se os tipos dos parâmetros estiverem incorretos
    """
    
    # Validações iniciais
    if not isinstance(df, pd.DataFrame):
        raise TypeError("O parâmetro 'df' deve ser um pandas DataFrame")
    
    if df.empty:
        raise ValueError("O DataFrame não pode estar vazio")
    
    if not isinstance(target_col, str):
        raise TypeError("O parâmetro 'target_col' deve ser uma string")
    
    if not isinstance(limite_categorico, int) or limite_categorico <= 0:
        raise ValueError("O parâmetro 'limite_categorico' deve ser um inteiro positivo")
    
    # Verifica se target_col existe no DataFrame
    if target_col not in df.columns:
        available_cols = ", ".join(df.columns.tolist()[:10])  # Mostra apenas primeiras 10
        suffix = "..." if len(df.columns) > 10 else ""
        raise ValueError(
            f"Coluna target '{target_col}' não encontrada no DataFrame. "
            f"Colunas disponíveis: {available_cols}{suffix}"
        )
    
    # Inicialização de variáveis
    num_cols = []
    cat_cols = []
    ignored_cols = []
    
    # Tratamento de parâmetros opcionais
    force_categorical = force_categorical or []
    id_patterns = id_patterns or ['client_id', '_id', 'id_', 'codigo', 'key']
    
    # Validação do force_categorical
    if not isinstance(force_categorical, list):
        raise TypeError("O parâmetro 'force_categorical' deve ser uma lista de strings")
    
    # Verifica se colunas em force_categorical existem
    missing_forced = [col for col in force_categorical if col not in df.columns]
    if missing_forced:
        warnings.warn(
            f"Colunas em force_categorical não encontradas: {missing_forced}",
            UserWarning
        )
        force_categorical = [col for col in force_categorical if col in df.columns]
    
    # Cria DataFrame sem a coluna target
    try:
        df_work = df.drop(columns=[target_col], errors='raise')
    except KeyError as e:
        raise ValueError(f"Erro ao remover coluna target: {e}")
    
    if verbose:
        print(f"Analisando {len(df_work.columns)} colunas (excluindo target '{target_col}')...")
        print("-" * 60)
    
    # Análise das colunas
    for col in df_work.columns:
        try:
            # Obter informações básicas da coluna
            tipo = df_work[col].dtype
            non_null_count = df_work[col].count()
            total_count = len(df_work)
            missing_pct = ((total_count - non_null_count) / total_count) * 100
            
            # Força colunas explicitamente marcadas como categóricas
            if col in force_categorical:
                cat_cols.append(col)
                if verbose:
                    print(f"✓ '{col}' -> CATEGÓRICA (forçada)")
                continue
            
            # Verifica se é coluna com muitos valores missing
            if missing_pct > 90:
                ignored_cols.append(col)
                if verbose:
                    print(f"⚠ '{col}' -> IGNORADA ({missing_pct:.1f}% valores ausentes)")
                continue
            
            # Classificação por tipo de dados
            if pd.api.types.is_numeric_dtype(tipo):
                # Verifica se é uma coluna ID numérica
                if remove_ids and _is_id_column(col, df_work[col], id_patterns):
                    ignored_cols.append(col)
                    if verbose:
                        print(f"🗑 '{col}' -> REMOVIDA (identificada como ID)")
                else:
                    num_cols.append(col)
                    if verbose:
                        unique_count = df_work[col].nunique(dropna=True)
                        print(f"📊 '{col}' -> NUMÉRICA ({unique_count} valores únicos)")
            
            elif tipo == 'object' or pd.api.types.is_string_dtype(tipo):
                # Remove IDs textuais se solicitado
                if remove_ids and _is_id_column(col, df_work[col], id_patterns):
                    ignored_cols.append(col)
                    if verbose:
                        print(f"🗑 '{col}' -> REMOVIDA (identificada como ID)")
                    continue
                
                unique_count = df_work[col].nunique(dropna=True)
                
                if unique_count <= limite_categorico:
                    cat_cols.append(col)
                    if verbose:
                        print(f"🏷 '{col}' -> CATEGÓRICA ({unique_count} categorias)")
                else:
                    ignored_cols.append(col)
                    if verbose:
                        print(f"⚠ '{col}' -> IGNORADA (muitas categorias: {unique_count})")
            
            elif pd.api.types.is_bool_dtype(tipo):
                cat_cols.append(col)
                if verbose:
                    print(f"☑ '{col}' -> CATEGÓRICA (booleana)")
            
            elif pd.api.types.is_datetime64_any_dtype(tipo):
                ignored_cols.append(col)
                if verbose:
                    print(f"📅 '{col}' -> IGNORADA (datetime)")
            
            else:
                ignored_cols.append(col)
                if verbose:
                    print(f"❓ '{col}' -> IGNORADA (tipo não suportado: {tipo})")
        
        except Exception as e:
            ignored_cols.append(col)
            if verbose:
                print(f"❌ '{col}' -> ERRO ao processar: {str(e)}")
            warnings.warn(f"Erro ao processar coluna '{col}': {str(e)}", UserWarning)
    
    # Remoção adicional de IDs se solicitado
    if remove_ids:
        num_cols, cat_cols = _remove_id_columns(num_cols, cat_cols, id_patterns, verbose)
    
    # Relatório final
    if verbose:
        print("\n" + "="*60)
        print("RESUMO DA CLASSIFICAÇÃO:")
        print("="*60)
        
        print(f"\n📊 VARIÁVEIS NUMÉRICAS ({len(num_cols)}):")
        if num_cols:
            for col in sorted(num_cols):
                print(f"   • {col}")
        else:
            print("   (nenhuma encontrada)")
        
        print(f"\n🏷 VARIÁVEIS CATEGÓRICAS ({len(cat_cols)}):")
        if cat_cols:
            for col in sorted(cat_cols):
                print(f"   • {col}")
        else:
            print("   (nenhuma encontrada)")
        
        if ignored_cols:
            print(f"\n⚠ COLUNAS IGNORADAS ({len(ignored_cols)}):")
            for col in sorted(ignored_cols):
                print(f"   • {col}")
        
        print(f"\n📈 ESTATÍSTICAS:")
        print(f"   • Total de colunas analisadas: {len(df_work.columns)}")
        print(f"   • Colunas numéricas: {len(num_cols)}")
        print(f"   • Colunas categóricas: {len(cat_cols)}")
        print(f"   • Colunas ignoradas: {len(ignored_cols)}")
        print(f"   • Taxa de utilização: {((len(num_cols) + len(cat_cols)) / len(df_work.columns) * 100):.1f}%")
    
    return num_cols, cat_cols


def _is_id_column(col_name: str, col_data: pd.Series, id_patterns: List[str]) -> bool:
    """
    Verifica se uma coluna é provavelmente um ID baseado no nome e características.
    
    Parâmetros:
    -----------
    col_name : str
        Nome da coluna
    col_data : pd.Series
        Dados da coluna
    id_patterns : List[str]
        Padrões para identificar IDs
    
    Retorna:
    --------
    bool
        True se a coluna for identificada como ID
    """
    col_lower = col_name.lower()
    
    # Verifica padrões no nome
    name_match = any(pattern.lower() in col_lower for pattern in id_patterns)
    
    # Verifica características dos dados
    unique_ratio = col_data.nunique() / len(col_data) if len(col_data) > 0 else 0
    high_uniqueness = unique_ratio > 0.95  # Mais de 95% de valores únicos
    
    return name_match or high_uniqueness


def _remove_id_columns(num_cols: List[str], cat_cols: List[str], 
                      id_patterns: List[str], verbose: bool) -> Tuple[List[str], List[str]]:
    """
    Remove colunas identificadas como IDs das listas de colunas numéricas e categóricas.
    
    Parâmetros:
    -----------
    num_cols : List[str]
        Lista de colunas numéricas
    cat_cols : List[str]
        Lista de colunas categóricas
    id_patterns : List[str]
        Padrões para identificar IDs
    verbose : bool
        Se True, imprime remoções
    
    Retorna:
    --------
    Tuple[List[str], List[str]]
        Tupla com listas atualizadas (num_cols, cat_cols)
    """
    original_num = len(num_cols)
    original_cat = len(cat_cols)
    
    # Remove IDs das colunas numéricas
    num_cols_filtered = []
    for col in num_cols:
        if not any(pattern.lower() in col.lower() for pattern in id_patterns):
            num_cols_filtered.append(col)
        elif verbose:
            print(f"🗑 Removendo '{col}' das numéricas (padrão ID detectado)")
    
    # Remove IDs das colunas categóricas
    cat_cols_filtered = []
    for col in cat_cols:
        if not any(pattern.lower() in col.lower() for pattern in id_patterns):
            cat_cols_filtered.append(col)
        elif verbose:
            print(f"🗑 Removendo '{col}' das categóricas (padrão ID detectado)")
    
    removed_count = (original_num + original_cat) - (len(num_cols_filtered) + len(cat_cols_filtered))
    if verbose and removed_count > 0:
        print(f"📋 Total de colunas ID removidas: {removed_count}")
    
    return num_cols_filtered, cat_cols_filtered