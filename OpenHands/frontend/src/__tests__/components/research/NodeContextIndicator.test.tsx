/**
 * Tests for NodeContextIndicator component
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { NodeContextIndicator } from '#/components/research/NodeContextIndicator';
import { useNodeEventStore } from '#/state/node-event-store';
import { useResearchTreeStore, ResearchNode } from '#/state/research-tree-store';

// Mock the stores
vi.mock('#/state/node-event-store');
vi.mock('#/state/research-tree-store');

describe('NodeContextIndicator', () => {
  const mockNode: ResearchNode = {
    id: 'node_1',
    type: 'web_search',
    title: 'Search for PostgreSQL documentation',
    content: 'Searching for information about PostgreSQL query optimization',
    status: 'running',
    visits: 5,
    prior: 0.8,
    avg_value: 0.75,
    cost: 100,
    tokens_used: 500,
    created_at: '2024-01-01T00:00:00Z',
    metadata: { conversation_id: 'conv_123' },
  };

  const mockSwitchToRootContext = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    
    // Default mock implementation - not in node context
    vi.mocked(useNodeEventStore).mockReturnValue('root' as any);
  });

  describe('Visibility', () => {
    it('should not render when not in node context', () => {
      vi.mocked(useNodeEventStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({
            contextMode: 'root',
            activeNodeId: null,
            switchToRootContext: mockSwitchToRootContext,
          });
        }
        return null;
      });

      vi.mocked(useResearchTreeStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({ nodes: new Map() });
        }
        return new Map();
      });

      const { container } = render(<NodeContextIndicator />);
      expect(container.firstChild).toBeNull();
    });

    it('should render when in node context with valid node', () => {
      vi.mocked(useNodeEventStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({
            contextMode: 'node',
            activeNodeId: 'node_1',
            switchToRootContext: mockSwitchToRootContext,
          });
        }
        return null;
      });

      const nodesMap = new Map([[mockNode.id, mockNode]]);
      vi.mocked(useResearchTreeStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({ nodes: nodesMap });
        }
        return nodesMap;
      });

      render(<NodeContextIndicator />);
      
      expect(screen.getByText('Search for PostgreSQL documentation')).toBeInTheDocument();
    });

    it('should not render when node is not found', () => {
      vi.mocked(useNodeEventStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({
            contextMode: 'node',
            activeNodeId: 'non_existent_node',
            switchToRootContext: mockSwitchToRootContext,
          });
        }
        return null;
      });

      vi.mocked(useResearchTreeStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({ nodes: new Map() });
        }
        return new Map();
      });

      const { container } = render(<NodeContextIndicator />);
      expect(container.firstChild).toBeNull();
    });
  });

  describe('Node Information Display', () => {
    beforeEach(() => {
      vi.mocked(useNodeEventStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({
            contextMode: 'node',
            activeNodeId: 'node_1',
            switchToRootContext: mockSwitchToRootContext,
          });
        }
        return null;
      });
    });

    it('should display node title', () => {
      const nodesMap = new Map([[mockNode.id, mockNode]]);
      vi.mocked(useResearchTreeStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({ nodes: nodesMap });
        }
        return nodesMap;
      });

      render(<NodeContextIndicator />);
      
      expect(screen.getByText('Search for PostgreSQL documentation')).toBeInTheDocument();
    });

    it('should display formatted node type', () => {
      const nodesMap = new Map([[mockNode.id, mockNode]]);
      vi.mocked(useResearchTreeStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({ nodes: nodesMap });
        }
        return nodesMap;
      });

      render(<NodeContextIndicator />);
      
      // web_search should become "Web Search"
      expect(screen.getByText('Web Search')).toBeInTheDocument();
    });

    it('should display node content when available', () => {
      const nodesMap = new Map([[mockNode.id, mockNode]]);
      vi.mocked(useResearchTreeStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({ nodes: nodesMap });
        }
        return nodesMap;
      });

      render(<NodeContextIndicator />);
      
      expect(screen.getByText('Searching for information about PostgreSQL query optimization')).toBeInTheDocument();
    });

    it('should not crash when node content is missing', () => {
      const nodeWithoutContent = { ...mockNode, content: '' };
      const nodesMap = new Map([[nodeWithoutContent.id, nodeWithoutContent]]);
      
      vi.mocked(useResearchTreeStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({ nodes: nodesMap });
        }
        return nodesMap;
      });

      render(<NodeContextIndicator />);
      
      expect(screen.getByText('Search for PostgreSQL documentation')).toBeInTheDocument();
      // Content should not be rendered when empty
      expect(screen.queryByText('Searching for information')).not.toBeInTheDocument();
    });
  });

  describe('Status Indicator', () => {
    beforeEach(() => {
      vi.mocked(useNodeEventStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({
            contextMode: 'node',
            activeNodeId: 'node_1',
            switchToRootContext: mockSwitchToRootContext,
          });
        }
        return null;
      });
    });

    it('should show running status with blue color', () => {
      const runningNode = { ...mockNode, status: 'running' };
      const nodesMap = new Map([[runningNode.id, runningNode]]);
      
      vi.mocked(useResearchTreeStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({ nodes: nodesMap });
        }
        return nodesMap;
      });

      const { container } = render(<NodeContextIndicator />);
      
      // Check for the Circle icon with blue color class
      const statusIndicator = container.querySelector('.text-blue-500');
      expect(statusIndicator).toBeInTheDocument();
    });

    it('should show complete status with green color', () => {
      const completeNode = { ...mockNode, status: 'complete' };
      const nodesMap = new Map([[completeNode.id, completeNode]]);
      
      vi.mocked(useResearchTreeStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({ nodes: nodesMap });
        }
        return nodesMap;
      });

      const { container } = render(<NodeContextIndicator />);
      
      const statusIndicator = container.querySelector('.text-green-500');
      expect(statusIndicator).toBeInTheDocument();
    });

    it('should show failed status with red color', () => {
      const failedNode = { ...mockNode, status: 'failed' };
      const nodesMap = new Map([[failedNode.id, failedNode]]);
      
      vi.mocked(useResearchTreeStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({ nodes: nodesMap });
        }
        return nodesMap;
      });

      const { container } = render(<NodeContextIndicator />);
      
      const statusIndicator = container.querySelector('.text-red-500');
      expect(statusIndicator).toBeInTheDocument();
    });

    it('should show pending status with yellow color', () => {
      const pendingNode = { ...mockNode, status: 'pending' };
      const nodesMap = new Map([[pendingNode.id, pendingNode]]);
      
      vi.mocked(useResearchTreeStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({ nodes: nodesMap });
        }
        return nodesMap;
      });

      const { container } = render(<NodeContextIndicator />);
      
      const statusIndicator = container.querySelector('.text-yellow-500');
      expect(statusIndicator).toBeInTheDocument();
    });
  });

  describe('User Interactions', () => {
    beforeEach(() => {
      vi.mocked(useNodeEventStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({
            contextMode: 'node',
            activeNodeId: 'node_1',
            switchToRootContext: mockSwitchToRootContext,
          });
        }
        return null;
      });

      const nodesMap = new Map([[mockNode.id, mockNode]]);
      vi.mocked(useResearchTreeStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({ nodes: nodesMap });
        }
        return nodesMap;
      });
    });

    it('should call switchToRootContext when Return button is clicked', async () => {
      const user = userEvent.setup();
      render(<NodeContextIndicator />);
      
      const returnButton = screen.getByRole('button', { name: /Return to Main Conversation/i });
      await user.click(returnButton);
      
      expect(mockSwitchToRootContext).toHaveBeenCalledTimes(1);
    });

    it('should call switchToRootContext when X button is clicked', async () => {
      const user = userEvent.setup();
      render(<NodeContextIndicator />);
      
      const closeButton = screen.getByRole('button', { name: /Close node context/i });
      await user.click(closeButton);
      
      expect(mockSwitchToRootContext).toHaveBeenCalledTimes(1);
    });

    it('should have accessible button labels', () => {
      render(<NodeContextIndicator />);
      
      expect(screen.getByRole('button', { name: /Return to Main Conversation/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Close node context/i })).toBeInTheDocument();
    });
  });

  describe('Type Label Formatting', () => {
    beforeEach(() => {
      vi.mocked(useNodeEventStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({
            contextMode: 'node',
            activeNodeId: 'node_1',
            switchToRootContext: mockSwitchToRootContext,
          });
        }
        return null;
      });
    });

    it('should format web_search as "Web Search"', () => {
      const node = { ...mockNode, type: 'web_search' };
      const nodesMap = new Map([[node.id, node]]);
      
      vi.mocked(useResearchTreeStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({ nodes: nodesMap });
        }
        return nodesMap;
      });

      render(<NodeContextIndicator />);
      expect(screen.getByText('Web Search')).toBeInTheDocument();
    });

    it('should format code_search as "Code Search"', () => {
      const node = { ...mockNode, type: 'code_search' };
      const nodesMap = new Map([[node.id, node]]);
      
      vi.mocked(useResearchTreeStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({ nodes: nodesMap });
        }
        return nodesMap;
      });

      render(<NodeContextIndicator />);
      expect(screen.getByText('Code Search')).toBeInTheDocument();
    });

    it('should format single word type as capitalized', () => {
      const node = { ...mockNode, type: 'hypothesis' };
      const nodesMap = new Map([[node.id, node]]);
      
      vi.mocked(useResearchTreeStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({ nodes: nodesMap });
        }
        return nodesMap;
      });

      render(<NodeContextIndicator />);
      expect(screen.getByText('Hypothesis')).toBeInTheDocument();
    });
  });

  describe('Styling and Layout', () => {
    beforeEach(() => {
      vi.mocked(useNodeEventStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({
            contextMode: 'node',
            activeNodeId: 'node_1',
            switchToRootContext: mockSwitchToRootContext,
          });
        }
        return null;
      });

      const nodesMap = new Map([[mockNode.id, mockNode]]);
      vi.mocked(useResearchTreeStore).mockImplementation((selector: any) => {
        if (typeof selector === 'function') {
          return selector({ nodes: nodesMap });
        }
        return nodesMap;
      });
    });

    it('should have fixed positioning class', () => {
      const { container } = render(<NodeContextIndicator />);
      
      const banner = container.querySelector('.fixed');
      expect(banner).toBeInTheDocument();
    });

    it('should have blue gradient background', () => {
      const { container } = render(<NodeContextIndicator />);
      
      const banner = container.querySelector('.bg-gradient-to-r');
      expect(banner).toBeInTheDocument();
    });

    it('should have proper z-index for overlay', () => {
      const { container } = render(<NodeContextIndicator />);
      
      const banner = container.querySelector('.z-40');
      expect(banner).toBeInTheDocument();
    });
  });
});
