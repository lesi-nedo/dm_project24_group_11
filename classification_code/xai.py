import json
import os
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Union
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from raiutils.exceptions import UserConfigValidationException

import numpy as np
import pandas as pd
import torch
import shap
import lime
import dice_ml
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from functools import partial



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger('shap').setLevel(logging.WARNING)

class ExplanationVisualizer:
    """Handles visualization of model explanations."""
    
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = save_dir
        plt.style.use("seaborn-v0_8")
    
    # Debug version of plot_shap_summary method
    def plot_shap_summary(self, explanations: Dict, X_test: np.ndarray, plot_type: str = "bar"):
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        """Plot SHAP summary visualization."""
        plt.figure(figsize=(12, 8))
        
        
        shap_values = np.array(explanations['shap_values'])
        
      
        
        
        if plot_type == "bar":
            shap.summary_plot(
                shap_values,
                X_test,
                feature_names=explanations['feature_names'],
                plot_type="bar",
                max_display=12,  # Ensure all features are shown
                show=False
            )
        else:
            shap.plots.violin(
                shap_values,
                features=X_test,
                feature_names=explanations['feature_names'],
                max_display=12,
                plot_type="layered_violin"
            )
        warnings.filterwarnings("default", category=UserWarning)
        plt.tight_layout()
        if self.save_dir:
            plt.savefig(f"{self.save_dir}/shap_summary_{plot_type}.png")
        return plt.gcf()

    def plot_feature_importance(self, explanations: Dict):
        """Plot global feature importance."""
        # Handle both single and multi-class SHAP values
        shap_values = explanations['shap_values']
        
                
        feat_importance = pd.DataFrame({
            'feature': explanations['feature_names'],
            'importance': np.abs(shap_values).mean(0)
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_importance, y='feature', x='importance')
        plt.title('Global Feature Importance')
        plt.tight_layout()
        if self.save_dir:
            plt.savefig(f"{self.save_dir}/feature_importance.png")
        return plt.gcf() 

    def plot_lime_explanation(self, lime_exp, top_k: int = 10):
        """Plot LIME explanation for a single prediction."""
        plt.figure(figsize=(10, 6))
        # Get the predicted class label
        pred_label = lime_exp.available_labels()[0]  # Use first available label
        lime_exp.as_pyplot_figure(label=pred_label)
        plt.tight_layout()
        if self.save_dir:
            plt.savefig(f"{self.save_dir}/lime_explanation.png")
        return plt.gcf()

    def plot_counterfactuals(self, explanations: Dict, cf_params:dict, instance_idx: int = 0, samples:int=1, X_test: Optional[np.ndarray] = None):
        """Plot counterfactual examples comparison."""
        if explanations['dice_explainer'] is None:
            raise ValueError("No counterfactual explainer found")
            
        # Use X_test from explanations if not provided separately
        X_test_data = X_test if X_test is not None else explanations.get('X_test')
        if X_test_data is None:
            raise ValueError("X_test data not found in explanations and not provided separately")
        X_test_df = None
        # Convert numpy array to pandas DataFrame if needed
        if isinstance(X_test_data, np.ndarray):
            X_test_df = pd.DataFrame(
                X_test_data[instance_idx:instance_idx+samples],
                columns=explanations['feature_names'],
                dtype=np.float32
            )
        try:  
            counterfactuals = explanations['dice_explainer'].generate_counterfactuals(
                X_test_df,
                **cf_params
            )
        
        
            # Add probability column
            if hasattr(counterfactuals, 'cf_examples_list'):
                for cf_example in counterfactuals.cf_examples_list:
                    probs = torch.sigmoid(torch.tensor(cf_example.new_outcome)).numpy()
                    cf_example.final_cfs_df['prediction'] = probs
            
            plt.figure(figsize=(12, 6))
            counterfactuals.visualize_as_dataframe(show_only_changes=True)
            plt.tight_layout()
            if self.save_dir:
                plt.savefig(f"{self.save_dir}/counterfactuals.png")
            return plt.gcf()
        except Exception as e:
                logger.warning(f"Counterfactual generation failed: {str(e)}")
                return None

class ModelWrapper:
    def __init__(self, model: Any):
        self.model = model
        self.model.eval()
        self.backend = "PYT"
        

    def __call__(self, X: torch.Tensor | np.ndarray) -> torch.Tensor:
        logger.info("Predicting probabilities...")
        if isinstance(X, np.ndarray):
            # Ensure all data is float32
            X = torch.FloatTensor(X.astype(np.float32))
        elif isinstance(X, pd.DataFrame):
            # Convert DataFrame to float32 numpy array
            X = torch.FloatTensor(X.values.astype(np.float32))

        with torch.no_grad():
            predictions = torch.sigmoid(self.model(X))

            return predictions
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Explicit probability prediction method"""
        return self(X).numpy()
    
    def predict_shap(self, X: np.ndarray) -> np.ndarray:
        """Convert model outputs to probabilities."""
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X.astype(np.float32))
            probs = torch.sigmoid(self.model(X)).numpy()
            # Return 2D array of probabilities [P(y=0), P(y=1)]
            return probs
        
    def predict_lime(self, X: np.ndarray) -> np.ndarray:
        """Convert model outputs to probabilities."""
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X.astype(np.float32))
            probs = torch.sigmoid(self.model(X)).numpy()
            # Return 2D array of probabilities [P(y=0), P(y=1)]
            return np.column_stack([1-probs, probs])
        

def get_model_explanations(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    class_names: List[str] = None,
    data_interface: Optional[dice_ml.Data] = None,
    n_samples: int = 100
) -> Dict[str, Any]:
    """
    Generate model explanations using multiple methods.
    
    Args:
        model: The trained model to explain
        X_train: Training data for the explainer
        X_test: Test instances to explain
        feature_names: List of feature names
        class_names: List of class names (optional)
        data_interface: DiCE data interface (optional)
        n_samples: Number of samples for SHAP
        
    Returns:
        Dictionary containing various explanations
    """
    try:
        # Wrap model to ensure consistent interface
        model_wrapper = ModelWrapper(model)
        
        logger.info("Generating SHAP explanations...")
        explainer = shap.KernelExplainer(
            model_wrapper.predict_shap,
            shap.sample(X_train, n_samples),
            output_names=class_names
        )
        shap_values = explainer.shap_values(
            X_test,
            show_progress=True
        )
        
        logger.info("Setting up LIME explainer...")
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=class_names or ['Class 0', 'Class 1'],
            mode='classification'
        )
        logger.info(f"Data interface: {data_interface}")
        dice_explainer = None
        if data_interface is not None:
            logger.info("Initializing DiCE explainer...")
            m = dice_ml.Model(model=model_wrapper.model, backend="PYT", model_type="classifier")
            dice_explainer = dice_ml.Dice(
                data_interface,
                m,
                method='random',
            )
       
        return {
            'shap_values': shap_values.squeeze(),
            'lime_explainer': lime_explainer,
            'dice_explainer': dice_explainer,
            'feature_names': feature_names,
            'class_names': class_names
        }
        
    except Exception as e:
        logger.error(f"Error generating explanations: {str(e)}")
        raise RuntimeError("Failed to generate model explanations") from e

class ExplanationOutput(TypedDict):
    lime: Any
    shap: Optional[np.ndarray]
    counterfactuals: Optional[Any]
    importance_scores: Dict[str, float]

def explain_prediction(
    explainers: Dict[str, Any],
    instance: np.ndarray,
    cf_params: dict,
    num_features: int = 10,
    counterfactual_params: Optional[Dict] = None
) -> ExplanationOutput:
    """Generate local explanations for a single instance.
    
    Args:
        explainers: Dictionary containing explainers:
            - lime_explainer: LIME explainer instance
            - shap_explainer: SHAP explainer instance (optional)
            - dice_explainer: DiCE explainer instance (optional)
        instance: Single instance to explain
        num_features: Number of top features to include
        counterfactual_params: Parameters for counterfactual generation
        
    Returns:
        ExplanationOutput containing:
            - lime: LIME explanation
            - shap: SHAP values (if available)
            - counterfactuals: Counterfactual examples (if available)
            - importance_scores: Feature importance scores
    
    Raises:
        ValueError: If required explainers are missing
        RuntimeError: If explanation generation fails
    """
    if not isinstance(instance, np.ndarray):
        instance = np.array(instance).reshape(1, -1)
    
    if 'lime_explainer' not in explainers:
        raise ValueError("LIME explainer required but not found")
        
    try:
        # Generate LIME explanation
        logger.info("Generating LIME explanation...")
        lime_exp = explainers['lime_explainer'].explain_instance(
            instance,
            lambda x: explainers['model_wrapper'].predict(x),
            num_features=num_features
        )
        
        # Generate SHAP values if available
        shap_values = None
        if 'shap_explainer' in explainers:
            logger.info("Generating SHAP values...")
            shap_values = explainers['shap_explainer'].shap_values(instance)
        
        # Generate counterfactuals if available
        counterfactuals = None
        if 'dice_explainer' in explainers:
            logger.info("Generating counterfactuals...")
            
            instance_df = pd.DataFrame(
                instance,
                columns=explainers['feature_names']
            )
            counterfactuals = explainers['dice_explainer'].generate_counterfactuals(
                instance_df,
                **cf_params
            )
        
        # Extract feature importance scores
        importance_scores = {
            feat: score for feat, score in 
            zip(lime_exp.feature_names, lime_exp.feature_importances)
        }
            
        return ExplanationOutput(
            lime=lime_exp,
            shap=shap_values,
            counterfactuals=counterfactuals,
            importance_scores=importance_scores
        )
        
    except Exception as e:
        logger.error(f"Error explaining prediction: {str(e)}")
        raise RuntimeError("Failed to explain prediction") from e

def analyze_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    X_test: np.ndarray,
    importance_metric: str = 'mean_abs',
    plot_params: Optional[Dict] = None,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze global feature importance using SHAP values.
    
    Args:
        shap_values: SHAP values from explainer
        feature_names: List of feature names
        X_test: Test data used for SHAP values
        importance_metric: 'mean_abs' or 'mean_squared'
        plot_params: Dictionary of plotting parameters
        save_path: Path to save plot
        
    Returns:
        DataFrame with feature importance scores
    """
    try:
        logger.info("Calculating feature importance...")
        
        # Calculate importance based on metric
        if importance_metric == 'mean_abs':
            importance = np.abs(shap_values).mean(0)
        elif importance_metric == 'mean_squared':
            importance = np.square(shap_values).mean(0)
        else:
            raise ValueError(f"Unknown importance metric: {importance_metric}")

        # Create importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # Plot with custom parameters
        plot_params = plot_params or {
            'figsize': (10, 6),
            'max_display': 20,
            'plot_type': 'bar',
            'show': True
        }
        
        plt.figure(figsize=plot_params['figsize'])
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=feature_names,
            max_display=plot_params['max_display'],
            plot_type=plot_params['plot_type'],
            show=plot_params['show']
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Plot saved to {save_path}")

        return feature_importance

    except Exception as e:
        logger.error(f"Error in feature importance analysis: {str(e)}")
        raise RuntimeError("Failed to analyze feature importance") from e


class RuleExtractionResult(TypedDict):
    rules_text: str
    rules_df: pd.DataFrame
    fidelity_score: float
    tree_model: DecisionTreeClassifier
    feature_importance: Dict[str, float]

def extract_rules(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    tree_params: Optional[Dict] = None,
    visualization_params: Optional[Dict] = None,
    output_format: str = 'all'
) -> RuleExtractionResult:
    """Extract interpretable rules from black-box model."""
    
    # Default parameters
    tree_params = tree_params or {
        'max_depth': 5,
        'min_samples_leaf': 10,
        'random_state': 42
    }
    
    # Separate figure and tree visualization parameters
    figure_params = {
        'figsize': (20, 10)
    }
    
    tree_viz_params = {
        'filled': True,
        'rounded': True,
        'fontsize': 10
    }

    if visualization_params:
        figure_params.update({k: v for k, v in visualization_params.items() 
                            if k in ['figsize']})
        tree_viz_params.update({k: v for k, v in visualization_params.items() 
                              if k in ['filled', 'rounded', 'fontsize']})

    try:
        model_wrapper = ModelWrapper(model)
        # Train tree approximation
        dt = DecisionTreeClassifier(**tree_params)
        predictions = (model_wrapper(X_train) > 0.5).int().numpy().flatten()
        dt.fit(X_train, predictions)
        dt_predictions = dt.predict(X_test)

        assert len(predictions) == len(dt_predictions), "Prediction lengths must match"


        # Extract rules in different formats
        rules_text = export_text(dt, 
                               feature_names=feature_names,
                               spacing=3)

        # Convert rules to DataFrame
        rules_df = _convert_rules_to_df(dt, feature_names)
        # Calculate fidelity
        fidelity_score = sklearn.metrics.accuracy_score(
            predictions,
            dt_predictions
        )

        # Get feature importance
        importance = dict(zip(feature_names, 
                            dt.feature_importances_))

        result = RuleExtractionResult(
            rules_text=rules_text,
            rules_df=rules_df,
            fidelity_score=fidelity_score,
            tree_model=dt,
            feature_importance=importance
        )

        # Generate visualizations if requested
        if output_format in ['all', 'plot']:
            fig = plt.figure(**figure_params)
            plot_tree(dt, 
                     feature_names=feature_names,
                     **tree_viz_params)
            result['tree_plot'] = fig

        return result

    except Exception as e:
        logger.error(f"Rule extraction failed: {str(e)}")
        raise RuntimeError("Failed to extract rules") from e

def _convert_rules_to_df(tree: DecisionTreeClassifier, 
                        feature_names: List[str]) -> pd.DataFrame:
    """Helper to convert tree rules to DataFrame format."""
    tree_rules = []
    n_nodes = tree.tree_.node_count
    
    def recurse(node, depth, path):
        if tree.tree_.feature[node] != sklearn.tree._tree.TREE_UNDEFINED:
            name = feature_names[tree.tree_.feature[node]]
            threshold = tree.tree_.threshold[node]
            
            left_path = path + [f"{name} <= {threshold:.2f}"]
            right_path = path + [f"{name} > {threshold:.2f}"]
            
            recurse(tree.tree_.children_left[node], depth + 1, left_path)
            recurse(tree.tree_.children_right[node], depth + 1, right_path)
        else:
            tree_rules.append({
                'rule': ' AND '.join(path),
                'prediction': tree.tree_.value[node].argmax(),
                'samples': tree.tree_.n_node_samples[node]
            })
            
    recurse(0, 1, [])
    return pd.DataFrame(tree_rules)


class CounterfactualResult(TypedDict):
    counterfactuals: Any  # DiCE counterfactual object
    proximity_scores: List[float]
    feature_changes: Dict[str, List[float]]
    diversity_score: float

def generate_counterfactuals(
    dice_explainer: Any,
    feature_names: Optional[List[str]],
    instance: np.ndarray | pd.DataFrame,
    cf_params: Optional[Dict] = None,
) -> CounterfactualResult:
    """Generate counterfactual explanations for a given instance."""
    try:
        
       
        cf_params = {**(cf_params or {})}

        if isinstance(instance, np.ndarray):
            
            instance = pd.DataFrame(
                instance,
                columns=feature_names,
                dtype=np.float32
            )
        
        try:

            counterfactuals = dice_explainer.generate_counterfactuals(
                instance,
                **cf_params
            )
        except Exception as e:
            logger.warning(f"Counterfactual generation failed: {str(e)}")
            return CounterfactualResult(
                counterfactuals=None,
                proximity_scores=[],
                feature_changes={},
                diversity_score=0.0
            )
             
        # Calculate metrics only if counterfactuals were generated
        proximity_scores = []
        feature_changes = {feat: [] for feat in feature_names}
        diversity_score = 0.0

        if hasattr(counterfactuals, 'cf_examples_list') and counterfactuals.cf_examples_list:
            cf_examples = counterfactuals.cf_examples_list[0].final_cfs_df
            original = instance.values if isinstance(instance, pd.DataFrame) else instance.reshape(1, -1)

            for cf in cf_examples.values:
                cf_features = cf[:-1]
                proximity_scores.append(np.linalg.norm(cf_features - original[0]))
                
                
                for feat_idx, feat_name in enumerate(feature_names):
                    feature_changes[feat_name].append(cf_features[feat_idx] - original[0][feat_idx])
            
            # Calculate diversity score if multiple counterfactuals exist
            if len(cf_examples) > 1:
                diversity_score = np.mean([
                    np.linalg.norm(cf1 - cf2)
                    for i, cf1 in enumerate(cf_examples.values)
                    for j, cf2 in enumerate(cf_examples.values)
                    if i < j
                ])

        return CounterfactualResult(
            counterfactuals=counterfactuals,
            proximity_scores=proximity_scores,
            feature_changes=feature_changes,
            diversity_score=diversity_score
        )
        
    except Exception as e:
        logger.error(f"Counterfactual generation failed: {str(e)}")
        raise
    
class ClusteringInsights(TypedDict):
    matching_patterns: List[Dict[str, Any]]
    new_patterns: List[Dict[str, Any]]
    unexpected_patterns: List[Dict[str, Any]]
    overlap_scores: Dict[str, float]
    visualization_data: Dict[str, Any]

def compare_with_clustering(
    explanations: Dict[str, Any],
    clustering_results: Dict[str, Any],
    threshold: float = 0.7
) -> ClusteringInsights:
    """Compare XAI explanations with clustering results."""
    
    insights = {
        'matching_patterns': [],
        'new_patterns': [],
        'unexpected_patterns': [],
        'overlap_scores': {},
        'visualization_data': {}
    }
    
    try:
        # Extract key metrics
        feature_importance = explanations.get('feature_importance', {})
        rules = explanations.get('rules_df', pd.DataFrame())
        counterfactuals = explanations.get('counterfactuals', None)
        cluster_centers = clustering_results.get('cluster_centers', [])
        cluster_labels = clustering_results.get('labels', [])
        
        # Compare feature importance with cluster characteristics
        for feature, importance in feature_importance.items():
            cluster_variance = np.var([
                center[feature] for center in cluster_centers
            ])
            correlation = np.corrcoef(importance, cluster_variance)[0,1]
            
            if correlation > threshold:
                insights['matching_patterns'].append({
                    'feature': feature,
                    'importance': importance,
                    'cluster_variance': cluster_variance,
                    'correlation': correlation
                })
            elif correlation < -threshold:
                insights['unexpected_patterns'].append({
                    'feature': feature,
                    'importance': importance,
                    'cluster_variance': cluster_variance,
                    'correlation': correlation
                })
        
        # Analyze rule boundaries vs cluster boundaries
        if not rules.empty:
            for _, rule in rules.iterrows():
                rule_bounds = _extract_rule_boundaries(rule['rule'])
                cluster_alignment = _check_cluster_alignment(
                    rule_bounds,
                    cluster_centers
                )
                if cluster_alignment > threshold:
                    insights['matching_patterns'].append({
                        'rule': rule['rule'],
                        'cluster_alignment': cluster_alignment
                    })
        
        # Analyze counterfactual transitions
        if counterfactuals is not None:
            for cf in counterfactuals:
                original_cluster = _get_nearest_cluster(
                    cf['original'],
                    cluster_centers
                )
                cf_cluster = _get_nearest_cluster(
                    cf['counterfactual'],
                    cluster_centers
                )
                if original_cluster != cf_cluster:
                    insights['new_patterns'].append({
                        'transition': f'Cluster {original_cluster} -> {cf_cluster}',
                        'features_changed': cf['changed_features']
                    })
        
        # Calculate overlap scores
        insights['overlap_scores'] = {
            'feature_importance': len(insights['matching_patterns']) / len(feature_importance),
            'rules': len([p for p in insights['matching_patterns'] if 'rule' in p]) / len(rules),
            'counterfactuals': len(insights['new_patterns']) / (len(counterfactuals) if counterfactuals else 1)
        }
        
        return insights
        
    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        raise RuntimeError("Failed to compare explanations with clustering") from e

def _extract_rule_boundaries(rule: str) -> Dict[str, Tuple[float, float]]:
    """Helper to extract numerical boundaries from rule text."""
    boundaries = {}
    conditions = rule.split(' AND ')
    for condition in conditions:
        if '<=' in condition:
            feature, value = condition.split('<=')
            boundaries[feature.strip()] = (-np.inf, float(value))
        elif '>' in condition:
            feature, value = condition.split('>')
            boundaries[feature.strip()] = (float(value), np.inf)
    return boundaries

def _check_cluster_alignment(
    rule_bounds: Dict[str, Tuple[float, float]],
    cluster_centers: List[np.ndarray]
) -> float:
    """Helper to check if rule boundaries align with cluster centroids."""
    alignments = []
    for center in cluster_centers:
        aligned_features = 0
        for feature, (lower, upper) in rule_bounds.items():
            if lower <= center[feature] <= upper:
                aligned_features += 1
        alignments.append(aligned_features / len(rule_bounds))
    return max(alignments)

def _get_nearest_cluster(
    point: np.ndarray,
    cluster_centers: List[np.ndarray]
) -> int:
    """Helper to find nearest cluster center."""
    distances = [
        np.linalg.norm(point - center)
        for center in cluster_centers
    ]
    return np.argmin(distances)

class ExplanationReport(TypedDict):
    feature_importance: Dict[str, Union[List[str], str, float]]
    rules: Dict[str, Union[List[str], int, float]]
    counterfactuals: Dict[str, Union[List[Dict], float]]
    comparison: Dict[str, List[Dict[str, Any]]]

def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to native Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
    return obj

def generate_explanation_report(
    all_results: Dict[str, Any],
    max_features: int = 10,
    max_rules: int = 5
) -> ExplanationReport:
    """Generate comprehensive explanation report."""
    try:
        report = ExplanationReport(
            feature_importance={
                'top_features': [],
                'interpretation': '',
                'fidelity': 0.0
            },
            rules={
                'key_rules': [],
                'complexity': 0,
                'fidelity': 0.0
            },
            counterfactuals={
                'examples': [],
                'plausibility': 0.0
            },
            comparison={
                'confirmed_patterns': [],
                'new_insights': [],
                'discrepancies': []
            }
        )

        # Process feature importance
        if 'feature_importance' in all_results:
            importance = all_results['feature_importance']
            sorted_features = sorted(
                importance,
                key=lambda x: abs(np.mean(x[1])) if isinstance(x[1], np.ndarray) else abs(x[1]),
                reverse=True
            )
            report['feature_importance'].update({
                'top_features': [
                    {
                        'feature': f,
                        'importance': convert_to_serializable(v)
                    }
                    for f, v in zip(all_results['feature_names'], sorted_features[:max_features])
                ],
                'interpretation': f"Top feature is {sorted_features[0][0]} with {abs(np.mean(sorted_features[0][1])):.2f} importance",
                'fidelity': float(all_results.get('importance_fidelity', 0.0))
            })

        # Process rules
        if 'rules' in all_results:
            rules = all_results['rules']
            if 'rules_df' in rules:
                rules_list = rules['rules_df'].to_dict('records')[:max_rules]
                report['rules'].update({
                    'key_rules': [
                        {
                            'rule': str(r['rule']),
                            'support': int(r['samples'])
                        }
                        for r in rules_list
                    ],
                    'complexity': int(len(rules['rules_df'])),
                    'fidelity': float(rules.get('fidelity_score', 0.0))
                })

        # Process counterfactuals
        if 'counterfactuals' in all_results:
            cfs = all_results['counterfactuals']
            report['counterfactuals'].update({
                'examples': [
                    {
                        'original': convert_to_serializable(cf['original']),
                        'counterfactual': convert_to_serializable(cf['counterfactual']),
                        'changes': convert_to_serializable(cf['changes'])
                    }
                    for cf in cfs.get('examples', [])
                ],
                'plausibility': float(np.mean(cfs.get('proximity_scores', [0])))
            })

        return report

    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise RuntimeError("Failed to generate explanation report") from e
    

def analyze_model(model, X_train, X_test, feature_names, save_dir="xai_results", cf_params:dict = None, instance_idx=None, data_interface=None):
    """Run complete XAI analysis pipeline."""
    instance_idx = np.random.randint(0, X_test.shape[0]) if instance_idx is None else instance_idx

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = ExplanationVisualizer(save_dir=save_dir)
    model_wrapper = ModelWrapper(model)
    
    # 1. Generate all explanations
    explanations = get_model_explanations(
        model=model,
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names,
        data_interface=data_interface
    )
    
    # 2. Extract rules
    rules = extract_rules(
        model=model,
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names
    )
    
    # 3. Generate counterfactuals for specific instance
    counterfactuals = generate_counterfactuals(
        dice_explainer=explanations['dice_explainer'],
        instance=X_test[instance_idx:instance_idx+1],
        feature_names=feature_names,
        cf_params=cf_params
    )
    
    # 4. Plot results
    
    # 4.1 SHAP Summary
    visualizer.plot_shap_summary(
        explanations=explanations,
        X_test=X_test,
        plot_type="shap"
    )
    
    # 4.2 Feature Importance
    visualizer.plot_feature_importance(
        explanations=explanations
    )

    # 4.3 LIME explanation for specific instance
    lime_exp = explanations['lime_explainer'].explain_instance(
        X_test[instance_idx],
        model_wrapper.predict_lime,
        num_features=len(feature_names),
        top_labels=1
    )
    visualizer.plot_lime_explanation(lime_exp)
    
    # 4.4 Counterfactuals
    visualizer.plot_counterfactuals(
        explanations=explanations,
        instance_idx=instance_idx,
        X_test=X_test,
        cf_params=cf_params
    )
    
    # 5. Generate comprehensive report
    report = generate_explanation_report({
        'feature_importance': explanations['shap_values'],
        'rules': rules,
        'counterfactuals': counterfactuals,
        'max_features': len(feature_names),
        'feature_names': feature_names,
    })
    
    
    try:
        with open(f"{save_dir}/xai_report.json", 'w') as f:
            # Convert all numpy arrays to lists before saving
            serializable_report = json.loads(
                json.dumps(report, default=convert_to_serializable)
            )
            json.dump(serializable_report, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save report: {str(e)}")
        
    return report