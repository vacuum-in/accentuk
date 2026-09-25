FROM golang:1.25-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -buildvcs=false -o /out/stress-api ./cmd/stress-api

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/stress-api /stress-api
USER nonroot:nonroot
ENTRYPOINT ["/stress-api"]

