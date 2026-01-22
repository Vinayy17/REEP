"use client"

import {
  Globe,
  Crown,
  Code,
  Laptop,
} from "lucide-react"

export function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 flex items-center justify-center relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0">
        <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-purple-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 bg-indigo-500/20 rounded-full blur-3xl animate-pulse delay-500"></div>
      </div>

      {/* Splash Content */}
      <div className="text-center z-10 space-y-10">
        {/* Logo */}
        <div className="relative">
          <div className="w-32 h-32 mx-auto bg-gradient-to-r from-purple-400 to-blue-400 rounded-3xl flex items-center justify-center shadow-2xl animate-bounce">
            <Globe className="h-16 w-16 text-white" />
          </div>
          <div className="absolute -top-2 -right-2 w-8 h-8 bg-yellow-400 rounded-full flex items-center justify-center animate-spin">
            <Crown className="h-4 w-4 text-yellow-800" />
          </div>
        </div>

        {/* Company Name */}
        <div className="space-y-3">
          <h1 className="text-6xl font-bold bg-gradient-to-r from-purple-200 via-blue-200 to-indigo-200 bg-clip-text text-transparent animate-pulse">
            OutranSystems
          </h1>
          <p className="text-xl text-purple-200 font-medium">
            Enterprise Resource Planning
          </p>
          <p className="text-lg text-blue-200">
            for Gokhale Bhandu Snacks
          </p>
        </div>

        {/* Team */}
        <div className="space-y-6">
          <p className="text-purple-300 text-sm uppercase tracking-wider">
            Co-Founders & Lead Developers
          </p>

          <div className="flex flex-wrap justify-center gap-10">
            {/* Founder 1 */}
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-2 shadow-lg">
                <Crown className="h-8 w-8 text-white" />
              </div>
              <p className="text-white font-semibold">Shubhankar Maurya</p>
              <p className="text-purple-300 text-sm">Co-Founder</p>
            </div>

            {/* Founder 2 */}
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-blue-600 rounded-full flex items-center justify-center mx-auto mb-2 shadow-lg">
                <Crown className="h-8 w-8 text-white" />
              </div>
              <p className="text-white font-semibold">Rishikesh Narala</p>
              <p className="text-blue-300 text-sm">Co-Founder</p>
            </div>

            {/* Dev 1 */}
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-full flex items-center justify-center mx-auto mb-2 shadow-lg">
                <Code className="h-8 w-8 text-white" />
              </div>
              <p className="text-white font-semibold">Parth Mishra</p>
              <p className="text-emerald-300 text-sm">Lead Developer</p>
            </div>

            {/* Dev 2 */}
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-orange-500 to-orange-600 rounded-full flex items-center justify-center mx-auto mb-2 shadow-lg">
                <Laptop className="h-8 w-8 text-white" />
              </div>
              <p className="text-white font-semibold">Vinay Mamidala</p>
              <p className="text-orange-300 text-sm">Lead Developer</p>
            </div>
          </div>
        </div>

        {/* Loading Animation */}
        <div className="space-y-3">
          <div className="flex justify-center space-x-2">
            <div className="w-3 h-3 bg-purple-400 rounded-full animate-bounce"></div>
            <div className="w-3 h-3 bg-blue-400 rounded-full animate-bounce delay-100"></div>
            <div className="w-3 h-3 bg-indigo-400 rounded-full animate-bounce delay-200"></div>
          </div>
          <p className="text-purple-200 text-sm animate-pulse">
            Initializing System...
          </p>
        </div>
      </div>
    </div>
  )
}
