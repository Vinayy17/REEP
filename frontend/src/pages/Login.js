"use client";

import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { toast } from "sonner";
import {
  Package,
  Globe,
  Crown,
  Code,
  Laptop,
} from "lucide-react";

const Login = () => {
  /* ================= SPLASH STATE ================= */
  const [showSplash, setShowSplash] = useState(true);

  /* ================= AUTH STATE ================= */
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);

  const { login, register } = useAuth();
  const navigate = useNavigate();

  /* ================= SPLASH TIMER ================= */
  useEffect(() => {
    const timer = setTimeout(() => {
      setShowSplash(false);
    }, 2500); // splash duration

    return () => clearTimeout(timer);
  }, []);

  /* ================= SPLASH SCREEN ================= */
  if (showSplash) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 flex items-center justify-center relative overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0">
          <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-purple-500/20 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse delay-1000" />
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

          {/* Company */}
          <div className="space-y-2">
            <h1 className="text-6xl font-bold bg-gradient-to-r from-purple-200 via-blue-200 to-indigo-200 bg-clip-text text-transparent animate-pulse">
              OutranSystems
            </h1>
            <p className="text-xl text-purple-200">
              Enterprise Resource Planning
            </p>
            <p className="text-lg text-blue-200">
              for RRide Garage
            </p>
          </div>

          {/* Team */}
          <div className="space-y-4">
            <p className="text-purple-300 text-sm uppercase tracking-wider">
              Co-Founders & Lead Developers
            </p>

            <div className="flex flex-wrap justify-center gap-8">
              <div className="text-center">
                <div className="w-14 h-14 bg-purple-600 rounded-full flex items-center justify-center mx-auto mb-1">
                  <Crown className="h-6 w-6 text-white" />
                </div>
                <p className="text-white font-medium">
                  Shubhankar Maurya
                </p>
                <p className="text-purple-300 text-xs">
                  Co-Founder
                </p>
              </div>

              <div className="text-center">
                <div className="w-14 h-14 bg-blue-600 rounded-full flex items-center justify-center mx-auto mb-1">
                  <Crown className="h-6 w-6 text-white" />
                </div>
                <p className="text-white font-medium">
                  Rishikesh Narala
                </p>
                <p className="text-blue-300 text-xs">
                  Co-Founder
                </p>
              </div>

              <div className="text-center">
                <div className="w-14 h-14 bg-emerald-600 rounded-full flex items-center justify-center mx-auto mb-1">
                  <Code className="h-6 w-6 text-white" />
                </div>
                <p className="text-white font-medium">
                  Parth Mishra
                </p>
                <p className="text-emerald-300 text-xs">
                  Lead Developer
                </p>
              </div>

              <div className="text-center">
                <div className="w-14 h-14 bg-orange-600 rounded-full flex items-center justify-center mx-auto mb-1">
                  <Laptop className="h-6 w-6 text-white" />
                </div>
                <p className="text-white font-medium">
                  Vinay Mamidala
                </p>
                <p className="text-orange-300 text-xs">
                  Lead Developer
                </p>
              </div>
            </div>
          </div>

          {/* Loader */}
          <div className="space-y-3">
            <div className="flex justify-center space-x-2">
              <div className="w-3 h-3 bg-purple-400 rounded-full animate-bounce" />
              <div className="w-3 h-3 bg-blue-400 rounded-full animate-bounce delay-100" />
              <div className="w-3 h-3 bg-indigo-400 rounded-full animate-bounce delay-200" />
            </div>
            <p className="text-purple-200 text-sm animate-pulse">
              Initializing System...
            </p>
          </div>
        </div>
      </div>
    );
  }

  /* ================= LOGIN / REGISTER ================= */
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      if (isLogin) {
        await login(email, password);
        toast.success("Welcome back!");
      } else {
        await register(email, password, name);
        toast.success("Account created successfully!");
      }
      navigate("/");
    } catch (error) {
      toast.error(error.response?.data?.detail || "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-background">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-primary mb-4">
            <Package className="w-8 h-8 text-primary-foreground" />
          </div>
          <h1 className="text-4xl font-bold mb-2">
            InvenTrack
          </h1>
          <p className="text-muted-foreground">
            Manage your inventory with ease
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-2xl">
              {isLogin ? "Welcome Back" : "Create Account"}
            </CardTitle>
            <CardDescription>
              {isLogin
                ? "Sign in to your account"
                : "Get started with InvenTrack"}
            </CardDescription>
          </CardHeader>

          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              {!isLogin && (
                <div>
                  <Label>Name</Label>
                  <Input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                  />
                </div>
              )}

              <div>
                <Label>Email</Label>
                <Input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>

              <div>
                <Label>Password</Label>
                <Input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>

              <Button
                type="submit"
                className="w-full"
                disabled={loading}
              >
                {loading
                  ? "Please wait..."
                  : isLogin
                  ? "Sign In"
                  : "Sign Up"}
              </Button>
            </form>

            <div className="mt-6 text-center">
              <button
                type="button"
                onClick={() => setIsLogin(!isLogin)}
                className="text-sm text-primary"
              >
                {isLogin
                  ? "Don't have an account? Sign up"
                  : "Already have an account? Sign in"}
              </button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Login;
