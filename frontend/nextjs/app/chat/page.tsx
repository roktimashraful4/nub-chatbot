'use client';

import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import { Send, Home, Bot, User } from 'lucide-react';
import { toast } from 'sonner';
import ReactMarkdown from "react-markdown";
import remarkBreaks from 'remark-breaks';
interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

export default function ChatPage() {
  // creaete unique id , use usemamo hook
  const [sectionId, setSectionId] = useState<string>('');
  
  useEffect(() => {
    const newSectionId = Date.now().toString();
    setSectionId(newSectionId);
    return () => {
      console.log("Component unmounted!");
    };
  }, []);
  
  console.log(sectionId);
  
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: "Hello! How can I help you today?",
      role: 'assistant',
      timestamp: new Date(),
    },
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const aIResponse = async (userMessage: string, session: string): Promise<string> => {
    
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
    const res = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: userMessage,
        session_id:session,
      }),
    })
    
    if (!res.ok) {
      toast.error('Error getting AI response');
      return '';
    }

    const data = await res.json();
    
    return `${data.answer}`;
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage.trim(),
      role: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const aiResponse = await aIResponse(inputMessage.trim(), sectionId);
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: aiResponse,
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error getting AI response:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: "Sorry, I encountered an error. Please try again.",
        role: 'assistant',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-black flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-900 bg-black/90 backdrop-blur-md sticky top-0 z-10">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14">
            <div className="flex items-center space-x-4">
              <Link 
                href="/" 
                className="flex items-center space-x-2 text-gray-400 hover:text-white transition-colors"
              >
                <Home className="w-4 h-4" />
                <span className="text-sm">Home</span>
              </Link>
              <div className="w-px h-4 bg-gray-800"></div>
              <div className="flex items-center space-x-2">
                <Bot className="w-4 h-4 text-white" />
                <span className="font-medium text-white text-sm">NUB AI Chat</span>
              </div>
            </div>
            <div className="text-xs text-gray-500">
              {messages.length - 1} messages | Session Id:  {sectionId}
            </div>
          </div>
        </div>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex items-start space-x-4 animate-fade-in ${
                  message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                }`}
              >
                <div className={`flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center ${
                  message.role === 'user' 
                    ? 'bg-white' 
                    : 'bg-gray-800'
                }`}>
                  {message.role === 'user' ? (
                    <User className="w-3 h-3 text-black" />
                  ) : (
                    <Bot className="w-3 h-3 text-white" />
                  )}
                </div>
                <div className={`flex-1 max-w-2xl ${
                  message.role === 'user' ? 'text-right' : 'text-left'
                }`}>
                  <div className={`inline-block p-3 rounded-xl text-sm ${
                    message.role === 'user'
                      ? 'bg-gray-800 text-black'
                      : 'bg-gray-900 text-gray-100 border border-gray-800'
                  }`}>
                    <div className="leading-relaxed text-white max-w-none"> {/* Tailwind typography plugin styles */}
                      <ReactMarkdown remarkPlugins={[remarkBreaks]}>
                        {message.content}
                      </ReactMarkdown>
                    </div>
                  </div>
                  <div className={`text-xs text-gray-500 mt-2 ${
                    message.role === 'user' ? 'text-right' : 'text-left'
                  }`}>
                    {message.timestamp.toLocaleTimeString([], { 
                      hour: '2-digit', 
                      minute: '2-digit' 
                    })}
                  </div>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="flex items-start space-x-4 animate-fade-in">
                <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gray-800 flex items-center justify-center">
                  <Bot className="w-3 h-3 text-white" />
                </div>
                <div className="flex-1">
                  <div className="inline-block p-3 rounded-xl bg-gray-900 border border-gray-800">
                    <div className="flex space-x-2">
                      <div className="w-1.5 h-1.5 bg-gray-600 rounded-full animate-pulse-slow"></div>
                      <div className="w-1.5 h-1.5 bg-gray-600 rounded-full animate-pulse-slow" style={{ animationDelay: '0.2s' }}></div>
                      <div className="w-1.5 h-1.5 bg-gray-600 rounded-full animate-pulse-slow" style={{ animationDelay: '0.4s' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="border-t border-gray-900 bg-black/90 backdrop-blur-md">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-end space-x-4">
            <div className="flex-1">
              <input
                ref={inputRef}
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
             
                placeholder="Type your message..."
                disabled={isLoading}
                className="w-full px-4 py-3 bg-gray-900 border border-gray-800 rounded-lg focus:ring-1 focus:ring-gray-700 focus:border-gray-700 outline-none text-white placeholder-gray-500 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
              />
            </div>
            <button
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || isLoading}
              className="p-3  disabled:cursor-not-allowed rounded-lg transition-colors duration-200 flex items-center justify-center"
            >
              <Send className={`w-7 h-7 ${!inputMessage.trim() || isLoading ? 'text-gray-500' : 'text-white'}`} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}