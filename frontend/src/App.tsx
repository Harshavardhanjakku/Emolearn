import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "next-themes";
import { SessionProvider } from "./contexts/session-context";
import { AppShell } from "./components/layout/AppShell";
import Achievements from "./pages/Achievements";
import Dashboard from "./pages/Dashboard";
import Gaming from "./pages/Gaming";
import Insights from "./pages/Insights";
import Index from "./pages/Index";
import Learning from "./pages/Learning";
import NotFound from "./pages/NotFound";
import Focus from "./pages/Focus";
import Break from "./pages/Break";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem storageKey="emolearn-theme">
      <SessionProvider>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <Routes>
              <Route element={<AppShell />}>
                <Route index element={<Dashboard />} />
                <Route path="dashboard" element={<Dashboard />} />
                <Route path="learning" element={<Learning />} />
                <Route path="gaming" element={<Gaming />} />
                <Route path="achievements" element={<Achievements />} />
                <Route path="insights" element={<Insights />} />
                <Route path="focus" element={<Focus />} />
                <Route path="break" element={<Break />} />
              </Route>
              <Route path="legacy" element={<Index />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </BrowserRouter>
        </TooltipProvider>
      </SessionProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;
