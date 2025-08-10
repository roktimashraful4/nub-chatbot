import Link from 'next/link';
import { MessageCircle, Zap, Shield, Sparkles } from 'lucide-react';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-black">
      {/* Navigation */}
      <nav className="border-b border-gray-900 bg-black/90 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14">
            <div className="flex items-center space-x-2">
              <MessageCircle className="w-6 h-6 text-white" />
              <span className="text-lg font-semibold text-white">NUB Chat</span>
            </div>
            <Link 
              href="/chat" 
              className="bg-white hover:bg-gray-100 text-black px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 hover:scale-105"
            >
              Start Chat
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-3xl mx-auto text-center">
          <div className="animate-fade-in">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold mb-6 leading-tight">
              <span className="text-white">Chat with</span>
              <span className="gradient-text block mt-1">
                NUB AI Assistant
              </span>
            </h1>
            <p className="text-lg text-gray-400 mb-8 max-w-xl mx-auto leading-relaxed">
              NUB AI Assistant is here to help you learn faster, work smarter, and find exactly what you need.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link 
                href="/chat" 
                className="bg-white hover:bg-gray-100 text-black px-6 py-3 rounded-lg font-medium transition-all duration-300 hover:scale-105"
              >
                Start Chat
              </Link>
              <button className="border border-gray-800 hover:border-gray-700 text-gray-300 px-6 py-3 rounded-lg font-medium transition-all duration-300 hover:bg-gray-900">
                Learn More
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-2xl sm:text-3xl font-bold mb-4 text-white">
              Why Choose Our <span className="gradient-text">AI</span>
            </h2>
            <p className="text-base text-gray-500 max-w-xl mx-auto">
              Powerful features for intelligent conversations.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-gray-950/50 backdrop-blur-sm p-6 rounded-xl border border-gray-900 hover:border-gray-800 transition-all duration-300 hover:transform hover:scale-105">
              <div className="w-10 h-10 bg-gray-800 rounded-lg flex items-center justify-center mb-4">
                <Zap className="w-5 h-5 text-white" />
              </div>
              <h3 className="text-lg font-semibold mb-3 text-white">Lightning Fast</h3>
              <p className="text-gray-500 text-sm leading-relaxed">
                Instant responses with advanced AI technology.
              </p>
            </div>

            <div className="bg-gray-950/50 backdrop-blur-sm p-6 rounded-xl border border-gray-900 hover:border-gray-800 transition-all duration-300 hover:transform hover:scale-105">
              <div className="w-10 h-10 bg-gray-800 rounded-lg flex items-center justify-center mb-4">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <h3 className="text-lg font-semibold mb-3 text-white">Creative & Smart</h3>
              <p className="text-gray-500 text-sm leading-relaxed">
                Intelligent responses for creative and analytical tasks.
              </p>
            </div>

            <div className="bg-gray-950/50 backdrop-blur-sm p-6 rounded-xl border border-gray-900 hover:border-gray-800 transition-all duration-300 hover:transform hover:scale-105">
              <div className="w-10 h-10 bg-gray-800 rounded-lg flex items-center justify-center mb-4">
                <Shield className="w-5 h-5 text-white" />
              </div>
              <h3 className="text-lg font-semibold mb-3 text-white">Safe & Secure</h3>
              <p className="text-gray-500 text-sm leading-relaxed">
                Private conversations with enterprise-grade security.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-gray-950/30">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-2xl sm:text-3xl font-bold mb-6 text-white">
            Ready to Start Your <span className="gradient-text">AI Conversation</span>?
          </h2>
          <p className="text-base text-gray-400 mb-8 max-w-xl mx-auto">
            Join thousands of users who are already experiencing the future of AI-powered conversations.
          </p>
          <Link 
            href="/chat" 
            className="inline-block bg-white hover:bg-gray-100 text-black px-8 py-3 rounded-lg font-medium transition-all duration-300 hover:scale-105"
          >
            Get Started
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-900 py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-5xl mx-auto text-center">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <MessageCircle className="w-5 h-5 text-white" />
            <span className="text-base font-medium text-white">AI Chat</span>
          </div>
          <p className="text-gray-600 text-sm">
            © 2025 AI Chat. Built with Next.js 15.
          </p>
        </div>
      </footer>
    </div>
  );
}