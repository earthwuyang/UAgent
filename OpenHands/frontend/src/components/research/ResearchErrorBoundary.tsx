import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ResearchErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ResearchTreeView Error:', error, errorInfo);
    this.setState({
      error,
      errorInfo,
    });
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex h-full items-center justify-center bg-slate-900 p-8">
          <div className="max-w-md space-y-4 text-center">
            <div className="flex justify-center">
              <AlertTriangle size={48} className="text-amber-500" />
            </div>
            <h3 className="text-lg font-semibold text-slate-100">
              Research Tree Error
            </h3>
            <p className="text-sm text-slate-400">
              An error occurred while rendering the research tree. This might be
              due to invalid data or a rendering issue.
            </p>
            {this.state.error && (
              <details className="rounded border border-slate-700 bg-slate-800 p-3 text-left text-xs">
                <summary className="cursor-pointer font-medium text-slate-300">
                  Error Details
                </summary>
                <pre className="mt-2 overflow-auto text-slate-400">
                  {this.state.error.toString()}
                </pre>
              </details>
            )}
            <button
              type="button"
              onClick={this.handleReset}
              className="inline-flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 transition-colors"
            >
              <RefreshCw size={16} />
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
