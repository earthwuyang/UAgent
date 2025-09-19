import React, { useState } from 'react'
import { Brain, Zap, Search, Code, FlaskConical, ArrowRight } from 'lucide-react'
import SmartResearchInput from './SmartResearchInput'
import ResearchResult from './ResearchResult'
import type { RouteAndExecuteResponse } from '../types/api'

const HomePage: React.FC = () => {
  const [currentResult, setCurrentResult] = useState<RouteAndExecuteResponse | null>(null)

  const handleResultReceived = (result: RouteAndExecuteResponse) => {
    setCurrentResult(result)
  }

  const features = [
    {
      icon: Search,
      title: 'Deep Research',
      description: 'Comprehensive multi-source research across web, academic, and technical sources',
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
    },
    {
      icon: Code,
      title: 'Code Research',
      description: 'Repository analysis, code understanding, and implementation patterns',
      color: 'text-green-600',
      bgColor: 'bg-green-100',
    },
    {
      icon: FlaskConical,
      title: 'Scientific Research',
      description: 'Experimental research with hypothesis testing and iterative refinement',
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
    },
  ]

  const examples = [
    {
      title: 'Market Research',
      query: 'What are the latest trends in renewable energy markets and policies?',
      engine: 'Deep Research',
    },
    {
      title: 'Code Analysis',
      query: 'Find Python libraries for transformer attention mechanisms',
      engine: 'Code Research',
    },
    {
      title: 'Scientific Study',
      query: 'Design experiments to test whether sparse attention patterns improve transformer efficiency',
      engine: 'Scientific Research',
    },
  ]

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <div className="text-center space-y-6">
        <div className="flex justify-center">
          <div className="p-4 bg-gradient-to-r from-primary-500 to-purple-600 rounded-2xl">
            <Brain className="h-12 w-12 text-white" />
          </div>
        </div>

        <div className="space-y-4">
          <h1 className="text-4xl font-bold text-gray-900 text-balance">
            Universal Research Agent
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto text-balance">
            Intelligent AI system that automatically routes your research requests to specialized engines.
            From comprehensive literature reviews to experimental code analysis.
          </p>
        </div>

        {/* Quick Stats */}
        <div className="flex justify-center space-x-8 text-sm text-gray-500">
          <div className="flex items-center space-x-2">
            <Zap className="h-4 w-4 text-yellow-500" />
            <span>AI-Powered Routing</span>
          </div>
          <div className="flex items-center space-x-2">
            <Search className="h-4 w-4 text-blue-500" />
            <span>Multi-Engine Coordination</span>
          </div>
          <div className="flex items-center space-x-2">
            <FlaskConical className="h-4 w-4 text-purple-500" />
            <span>Scientific Workflows</span>
          </div>
        </div>
      </div>

      {/* Research Input */}
      <div className="max-w-4xl mx-auto">
        <SmartResearchInput onResultReceived={handleResultReceived} />
      </div>

      {/* Current Result */}
      {currentResult && (
        <div className="max-w-6xl mx-auto">
          <ResearchResult result={currentResult} />
        </div>
      )}

      {/* Features Grid */}
      <div className="grid md:grid-cols-3 gap-8">
        {features.map((feature, index) => (
          <div key={index} className="card p-6 hover:shadow-md transition-shadow">
            <div className="flex items-center space-x-4 mb-4">
              <div className={`p-3 rounded-lg ${feature.bgColor}`}>
                <feature.icon className={`h-6 w-6 ${feature.color}`} />
              </div>
              <h3 className="text-lg font-semibold text-gray-900">{feature.title}</h3>
            </div>
            <p className="text-gray-600">{feature.description}</p>
          </div>
        ))}
      </div>

      {/* Example Queries */}
      <div className="bg-white rounded-xl border border-gray-200 p-8">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Try These Examples</h2>
          <p className="text-gray-600">Click any example to see how smart routing works</p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {examples.map((example, index) => (
            <div
              key={index}
              className="cursor-pointer group bg-gray-50 hover:bg-gray-100 rounded-lg p-4 transition-colors"
              onClick={() => {
                // You could implement auto-filling the input here
                const event = new CustomEvent('fillQuery', { detail: example.query })
                window.dispatchEvent(event)
              }}
            >
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-medium text-gray-900">{example.title}</h4>
                <ArrowRight className="h-4 w-4 text-gray-400 group-hover:text-gray-600 transition-colors" />
              </div>
              <p className="text-sm text-gray-600 mb-3">"{example.query}"</p>
              <span className="inline-block px-2 py-1 text-xs font-medium bg-primary-100 text-primary-700 rounded-full">
                → {example.engine}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* How it Works */}
      <div className="text-center space-y-8">
        <h2 className="text-2xl font-bold text-gray-900">How Smart Routing Works</h2>

        <div className="grid md:grid-cols-4 gap-8">
          {[
            { step: '1', title: 'Submit Query', desc: 'Enter your research question or request' },
            { step: '2', title: 'AI Analysis', desc: 'LLM analyzes complexity and requirements' },
            { step: '3', title: 'Smart Routing', desc: 'Automatically routes to best engine' },
            { step: '4', title: 'Execution', desc: 'Coordinated multi-engine research workflow' },
          ].map((step, index) => (
            <div key={index} className="space-y-4">
              <div className="flex justify-center">
                <div className="w-12 h-12 bg-primary-600 text-white rounded-full flex items-center justify-center font-bold">
                  {step.step}
                </div>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">{step.title}</h3>
                <p className="text-sm text-gray-600">{step.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default HomePage