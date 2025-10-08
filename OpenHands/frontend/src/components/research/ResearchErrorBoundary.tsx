import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { cn } from '#/utils/utils';

interface ResearchErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onReset?: () => void;
}

interface ResearchErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  showDetails: boolean;
}

class ResearchErrorBoundary extends Component<
  ResearchErrorBoundaryProps,
  ResearchErrorBoundaryState
> {
  constructor(props: ResearchErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
    };
  }

  static getDerivedStateFromError(error: Error): ResearchErrorBoundaryState {
    return {
      hasError: true,
      error,
      errorInfo: null,
      showDetails: false,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('[Research Tree] Render error:', error, errorInfo);
    this.setState({ errorInfo });
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null, errorInfo: null, showDetails: false });
    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  toggleDetails = () => {
    this.setState((state) => ({ showDetails: !state.showDetails }));
  };

  render() {
    const { hasError, error, errorInfo, showDetails } = this.state;
    const { children, fallback } = this.props;

    if (!hasError) {
      return children;
    }

    if (fallback) {
      return fallback;
    }

    return (
      <div className="research-error-boundary">
        <AlertTriangle className="error-icon" size={64} aria-hidden />
        <h3>Something went wrong with the research tree</h3>
        {error?.message && <p className="error-message">{error.message}</p>}
        <button type="button" className="retry-button" onClick={this.handleRetry}>
          <RefreshCw size={16} />
          <span>Try Again</span>
        </button>
        {errorInfo && (
          <button
            type="button"
            className="details-toggle"
            onClick={this.toggleDetails}
          >
            {showDetails ? 'Hide details' : 'Show details'}
          </button>
        )}
        {showDetails && errorInfo && (
          <pre className={cn('error-stack', 'custom-scrollbar')}>
            {errorInfo.componentStack}
          </pre>
        )}
      </div>
    );
  }
}

ResearchErrorBoundary.displayName = 'ResearchErrorBoundary';

export { ResearchErrorBoundary };
