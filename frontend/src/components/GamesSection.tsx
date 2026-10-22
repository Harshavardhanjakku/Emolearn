import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Gamepad2, ArrowLeft, ExternalLink, Play, Filter, AlertCircle, RefreshCw } from 'lucide-react';
import { games, Game } from '@/config/games';
import { getGameUrl } from '@/config/gameServer';

interface GamesSectionProps {
  onClose: () => void;
}

export const GamesSection = ({ onClose }: GamesSectionProps) => {
  const [selectedGame, setSelectedGame] = useState<Game | null>(null);
  const [selectedEmotion, setSelectedEmotion] = useState<string>('all');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const emotions = [
    { id: 'all', name: 'All Games', icon: '🎮', color: 'from-gray-400 to-gray-600' },
    { id: 'Boredom', name: 'Boredom', icon: '😴', color: 'from-blue-400 to-blue-600' },
    { id: 'Confusion', name: 'Confusion', icon: '🤔', color: 'from-yellow-400 to-yellow-600' },
    { id: 'Fatigue', name: 'Fatigue', icon: '😴', color: 'from-green-400 to-green-600' },
    { id: 'Frustration', name: 'Frustration', icon: '😤', color: 'from-red-400 to-red-600' }
  ];

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Easy':
        return 'border-success/30 bg-success/10 text-success';
      case 'Medium':
        return 'border-warning/30 bg-warning/10 text-warning';
      case 'Hard':
        return 'border-destructive/30 bg-destructive/10 text-destructive';
      default:
        return 'border-border bg-muted text-muted-foreground';
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'Creative':
        return 'from-purple-400 to-purple-600';
      case 'Programming':
        return 'from-blue-400 to-blue-600';
      case 'Puzzle':
        return 'from-orange-400 to-orange-600';
      case 'Reflex':
        return 'from-green-400 to-green-600';
      case 'Energy':
        return 'from-pink-400 to-pink-600';
      case 'Trivia':
        return 'from-indigo-400 to-indigo-600';
      default:
        return 'from-gray-400 to-gray-600';
    }
  };

  const getEmotionIcon = (emotion: string) => {
    switch (emotion) {
      case 'Boredom':
        return '😴';
      case 'Confusion':
        return '🤔';
      case 'Fatigue':
        return '😴';
      case 'Frustration':
        return '😤';
      default:
        return '🎮';
    }
  };

  const getGameIcon = (gameId: string) => {
    switch (gameId) {
      // Boredom Games
      case 'emoji-doodle':
        return '🎨';
      case 'story-spinner':
        return '📖';
      case 'puzzle-pop':
        return '🧩';
      
      // Confusion Games
      case 'line-shuffle':
        return '🔀';
      case 'concept-match':
        return '🔗';
      case 'time-to-solve':
        return '⏰';
      
      // Fatigue Games
      case 'quick-match':
        return '⚡';
      case 'reflex-dash':
        return '🏃';
      case 'juice-it-up':
        return '🥤';
      
      // Frustration Games
      case 'smash-the-bug':
        return '🐛';
      case 'laugh-trivia':
        return '😄';
      case 'arrange-the-code':
        return '📝';
      
      default:
        return '🎮';
    }
  };

  const handlePlayGame = async (game: Game) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Use the configured game server URL
      const fullUrl = getGameUrl(game.gameUrl);
      
      // Test if the game server is accessible
      const response = await fetch(fullUrl, { method: 'HEAD' });
      
      if (response.ok) {
        window.open(fullUrl, '_blank');
      } else {
        throw new Error('Game server not responding');
      }
    } catch (err) {
      setError(`Unable to load ${game.title}. Please make sure the game server is running on port 8000.`);
      console.error('Game loading error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const filteredGames = selectedEmotion === 'all' 
    ? games 
    : games.filter(game => game.emotion === selectedEmotion);

  if (selectedGame) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary">
              <Gamepad2 className="w-4 h-4 text-white" />
            </div>
            <h3 className="text-lg font-semibold">{selectedGame.title}</h3>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSelectedGame(null)}
            className="text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="w-4 h-4 mr-1" />
            Back to Games
          </Button>
        </div>

        <Card className="border-border/70 bg-card/80 p-6 shadow-sm">
          <div className="space-y-4">
            <div className="text-center">
              <div className="flex justify-center mb-3">
                <span className="text-4xl">{getEmotionIcon(selectedGame.emotion)}</span>
              </div>
              <h4 className="mb-2 text-xl font-semibold text-foreground">{selectedGame.title}</h4>
              <p className="mb-4 text-muted-foreground">{selectedGame.description}</p>
              
              <div className="flex justify-center space-x-4 mb-6">
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${getDifficultyColor(selectedGame.difficulty)}`}>
                  {selectedGame.difficulty}
                </span>
                <span className="rounded-full border border-border bg-muted px-3 py-1 text-xs font-medium text-muted-foreground">
                  {selectedGame.category}
                </span>
                <span className="rounded-full border border-primary/25 bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                  {selectedGame.emotion}
                </span>
              </div>
            </div>

            {error && (
              <div className="mb-4 rounded-md border border-destructive/30 bg-destructive/10 p-4">
                <div className="flex items-center space-x-2">
                  <AlertCircle className="h-5 w-5 text-destructive" />
                  <p className="text-sm text-destructive">{error}</p>
                </div>
                <div className="mt-3">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePlayGame(selectedGame)}
                    className="border-destructive/30 text-destructive hover:bg-destructive/10"
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Try Again
                  </Button>
                </div>
              </div>
            )}

            <div className="flex space-x-3">
              <Button
                onClick={() => handlePlayGame(selectedGame)}
                disabled={isLoading}
                className="flex-1 bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                {isLoading ? (
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Play className="w-4 h-4 mr-2" />
                )}
                {isLoading ? 'Loading...' : 'Play Game'}
              </Button>
              <Button
                variant="outline"
                onClick={() => handlePlayGame(selectedGame)}
                disabled={isLoading}
              >
                <ExternalLink className="w-4 h-4 mr-2" />
                Open in New Tab
              </Button>
            </div>

            <div className="text-center pt-4">
              <p className="text-xs text-muted-foreground">
                Make sure the backend server is running on port 8000
              </p>
            </div>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-background/95 py-3 backdrop-blur-md">
        <div className="flex items-center space-x-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary">
            <Gamepad2 className="w-4 h-4 text-white" />
          </div>
          <h3 className="text-lg font-semibold">EmoLearn Games</h3>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={onClose}
          className="text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="w-4 h-4 mr-1" />
          Back to Assistant
        </Button>
      </div>

      {/* Emotion Filter */}
      <div className="flex flex-wrap gap-2 mb-4">
        {emotions.map((emotion) => (
          <Button
            key={emotion.id}
            variant={selectedEmotion === emotion.id ? "default" : "outline"}
            size="sm"
            onClick={() => setSelectedEmotion(emotion.id)}
            className={`flex items-center space-x-2 ${
              selectedEmotion === emotion.id 
                ? `bg-gradient-to-r ${emotion.color} text-white border-0` 
                : 'hover:bg-muted'
            }`}
          >
            <span className="text-lg">{emotion.icon}</span>
            <span>{emotion.name}</span>
          </Button>
        ))}
      </div>

      {error && (
        <div className="mb-4 rounded-md border border-warning/30 bg-warning/10 p-4">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-warning" />
            <p className="text-sm text-warning">
              {error}
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3">
        {filteredGames.map((game) => (
          <Card
            key={game.id}
            onClick={() => setSelectedGame(game)}
            className="cursor-pointer border-border/70 bg-card/80 p-4 shadow-sm transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md"
          >
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className={`w-10 h-10 bg-gradient-to-r ${getCategoryColor(game.category)} rounded-full flex items-center justify-center`}>
                  <span className="text-lg">{getGameIcon(game.id)}</span>
                </div>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getDifficultyColor(game.difficulty)}`}>
                  {game.difficulty}
                </span>
              </div>
              
              <div className="text-left">
                <h4 className="mb-1 text-sm font-medium text-foreground">{game.title}</h4>
                <p className="mb-2 line-clamp-2 text-xs text-muted-foreground">{game.description}</p>
                
                <div className="flex justify-between items-center">
                  <span className="rounded-full border border-border bg-muted px-2 py-1 text-xs font-medium text-muted-foreground">
                    {game.category}
                  </span>
                  <span className="rounded-full border border-primary/25 bg-primary/10 px-2 py-1 text-xs font-medium text-primary">
                    {game.emotion}
                  </span>
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>

      <div className="text-center pt-4">
        <p className="text-xs text-muted-foreground">
          Games designed for different emotional states to enhance learning experience
        </p>
        <p className="mt-1 text-xs text-muted-foreground/70">
          Total: {filteredGames.length} games • Filtered by: {selectedEmotion === 'all' ? 'All Emotions' : selectedEmotion}
        </p>
        <p className="mt-2 text-xs text-primary">
          💡 Tip: Make sure the backend server is running on port 8000
        </p>
      </div>
    </div>
  );
}; 